"""
model.py — U-Net

网络要做的任务：给一张加噪图 x_t 和当前步 t，预测加进去的噪声 ε。
数据流（对照架构图，28×28 单通道为例）：
    输入 (B,1,28,28)
      → 输入卷积            (B,64,28,28)
      → 编码器下采样 ×DEPTH (每步先把特征存一份作 skip)
          28→14 (通道64→128)，14→7 (通道128→256)
      → bottleneck（含自注意力）
      → 解码器上采样 ×DEPTH（先上采样，再和对应的 skip 拼接）
          7→14 (拼接后通道256→128)，14→28 (拼接后通道128→64)
      → 输出卷积            (B,1,28,28)  ← 即预测噪声

时间 t 通过 TimeEmbedding 变成向量，注入每个 ResidualBlock（FiLM：缩放+平移）。

💡 可单独测试本文件：
    python model.py
    会用随机张量跑一遍 forward，断言输入输出同形状、且反向传播有梯度。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import BASE_CHANNELS, DEPTH, ATTN_AT, TIME_EMB_DIM, DEVICE

class TimeEmbedding(nn.Module):
    """把整数时间步 t 变成连续向量（Transformer 式正弦编码 + 两层 MLP）。

    输入 : t    形状 (B,) 的整数
    输出 : emb  形状 (B, TIME_EMB_DIM)
    作用 : 让网络"知道现在是第几步"，这是它学出去噪的前提。
    """

    def __init__(self, dim=TIME_EMB_DIM):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device)
            * (-torch.log(torch.tensor(10000.0)) / half)
        )
        args = t[:, None].float() * freqs[None, :]          # (B, half)
        pe = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
        return self.mlp(pe)


class ResidualBlock(nn.Module):
    """基础块：GroupNorm→SiLU→Conv →(+时间)→ GroupNorm→SiLU→Conv，最后加残差。

    输入 : x (B,C,H,W)，t_emb (B, TIME_EMB_DIM)
    输出 : (B, out_ch, H, W)   （不改变空间尺寸）
    时间注入：t_emb 过一个小线性得到 (scale, shift)，对特征做 FiLM 调制。
    """

    def __init__(self, in_ch, out_ch, t_dim=TIME_EMB_DIM):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(min(8, in_ch), in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_dim, out_ch * 2),   # 输出 scale 和 shift
        )
        # 残差连接需要通道对齐：不同就 1×1 卷积
        self.resample = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x, t_emb):
        h = self.block1(x)
        scale, shift = self.time_emb(t_emb).chunk(2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        h = h * (1 + scale) + shift          # FiLM 调制
        h = self.block2(h)
        return h + self.resample(x)


class DownBlock(nn.Module):
    """编码器一层：若干 ResBlock + 一次下采样(stride=2 卷积)。"""

    def __init__(self, in_ch, out_ch, num_res=2):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(in_ch if i == 0 else out_ch, out_ch) for i in range(num_res)]
        )
        self.downsample = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, t_emb):
        for rb in self.res_blocks:
            x = rb(x, t_emb)
        skip = x                        # 存一份给解码器
        x = self.downsample(x)         # 空间尺寸减半
        return x, skip


class UpBlock(nn.Module):
    """解码器一层：上采样 + 拼接 skip + 若干 ResBlock。"""

    def __init__(self, in_ch, out_ch, skip_ch, num_res=2):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        # 拼接后通道 = in_ch + skip_ch，第一层 ResBlock 输入要对应
        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock(in_ch + skip_ch if i == 0 else out_ch, out_ch)
                for i in range(num_res)
            ]
        )

    def forward(self, x, skip, t_emb):
        x = self.upsample(x)                       # 空间尺寸翻倍
        if x.shape != skip.shape:                 # 防奇偶差 1 像素
            x = F.interpolate(x, size=skip.shape[-2:])
        x = torch.cat([x, skip], dim=1)           # 跳跃连接拼接
        for rb in self.res_blocks:
            x = rb(x, t_emb)
        return x


class Attention(nn.Module):
    """bottleneck 自注意力：捕获长程结构（手写数字的笔画）。单头简化版。"""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = torch.softmax(q.transpose(1, 2) @ k / (C ** 0.5), dim=-1)
        out = attn @ v.transpose(1, 2)            # (B, C, H*W)
        out = out.reshape(B, C, H, W)
        return x + self.proj(out)


class UNet(nn.Module):
    """完整 U-Net。把上面零件按"编码器—bottleneck—解码器"组装起来。"""

    def __init__(self, in_ch=1, base=BASE_CHANNELS, depth=DEPTH, attn_at=ATTN_AT):
        super().__init__()
        self.time_emb = TimeEmbedding()
        # 每一层通道数逐级翻倍：例如 [64, 128, 256]
        chs = [base * (2 ** i) for i in range(depth + 1)]

        self.in_conv = nn.Conv2d(in_ch, chs[0], 3, padding=1)

        self.downs = nn.ModuleList()
        for i in range(depth):
            self.downs.append(DownBlock(chs[i], chs[i + 1]))

        mid_layers = [ResidualBlock(chs[-1], chs[-1])]
        if attn_at >= 0:
            mid_layers.append(Attention(chs[-1]))
        mid_layers.append(ResidualBlock(chs[-1], chs[-1]))
        self.mid = nn.Sequential(*mid_layers)

        self.ups = nn.ModuleList()
        for i in range(depth):
            in_ch_i = chs[depth - i]               # 当前解码层输入通道
            out_ch_i = chs[depth - i - 1]          # 输出通道（上一层）
            skip_ch = chs[depth - i]               # 对应编码器存的 skip 通道
            self.ups.append(UpBlock(in_ch_i, out_ch_i, skip_ch))

        self.out_conv = nn.Sequential(
            nn.GroupNorm(min(8, chs[0]), chs[0]),
            nn.SiLU(),
            nn.Conv2d(chs[0], in_ch, 3, padding=1),
        )

    def forward(self, x, t):
        t_emb = self.time_emb(t)
        x = self.in_conv(x)
        skips = []
        for down in self.downs:
            x, skip = down(x, t_emb)
            skips.append(skip)
        x = self.mid(x)
        for up in self.ups:
            skip = skips.pop()
            x = up(x, skip, t_emb)
        return self.out_conv(x)


def test_model():
    """自检：随机张量跑 forward，断言输入输出同形状、且能反向传播。"""
    B = 4
    x = torch.randn(B, 1, 28, 28)
    t = torch.randint(0, 1000, (B,))
    model = UNet().to(DEVICE)
    out = model(x.to(DEVICE), t.to(DEVICE))
    assert out.shape == x.shape, f"输出 {tuple(out.shape)} != 输入 {tuple(x.shape)}"
    out.sum().backward()                          # 确认梯度能流回去
    print("[model] 输出形状:", tuple(out.shape), "— 前向+反向通过 ✅")


if __name__ == "__main__":
    test_model()

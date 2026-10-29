import torch
import torch.nn as nn


def sinusoidal_embedding(t, dim):
    """把整数时间步 t (B,) 变成正弦位置编码 (B, dim)。
    和 Transformer 里的位置编码是同一套思路。
    """
    device = t.device
    freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))  # (dim/2,)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)                               # (B, dim/2)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)                       # (B, dim)


class TimeEmbedding(nn.Module):
    """第3步零件①：时间嵌入。让网络知道"当前是第几步"。"""
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        return self.mlp(sinusoidal_embedding(t, self.mlp[0].in_features))


class ResBlock(nn.Module):
    """第3步零件②：残差块。time 通过"相加"方式注入每个块。"""
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        self.time_mlp = nn.Linear(time_dim, out_ch)            # 把 time 映射成 (B, out_ch)
        # 输入/输出通道不同就用 1x1 卷积对齐残差
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(self.conv1(x)))
        t = self.time_mlp(self.act(t_emb)).unsqueeze(-1).unsqueeze(-1)  # (B, out_ch, 1, 1)
        h = h + t                                                 # time 注入
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.shortcut(x)                               # 残差连接


class SelfAttention(nn.Module):
    """第3步零件③：自注意力（捕捉长程结构，比如数字笔画）。"""
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.attn = nn.MultiheadAttention(ch, num_heads=4, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.norm(x).flatten(2).transpose(1, 2)              # (B, H*W, C)
        y, _ = self.attn(y, y, y)
        return x + y.transpose(1, 2).reshape(b, c, h, w)


class UNet(nn.Module):
    """第3步主体：编码-解码对称结构 + 跳跃连接。
    输入 x_t，输出同尺寸的噪声预测 ε̂。
    """
    def __init__(self, in_ch=1, base=64, time_dim=256):
        super().__init__()
        self.time_embed = TimeEmbedding(time_dim)
        # 下采样路径
        self.enc1_1 = ResBlock(in_ch, base, time_dim)
        self.enc1_2 = ResBlock(base, base, time_dim)
        self.down1 = nn.Conv2d(base, base, 4, stride=2, padding=1)        # 28→14
        self.enc2_1 = ResBlock(base, base * 2, time_dim)
        self.enc2_2 = ResBlock(base * 2, base * 2, time_dim)
        self.down2 = nn.Conv2d(base * 2, base * 2, 4, stride=2, padding=1)  # 14→7
        self.enc3_1 = ResBlock(base * 2, base * 4, time_dim)
        self.enc3_2 = ResBlock(base * 4, base * 4, time_dim)
        # bottleneck
        self.attn = SelfAttention(base * 4)
        self.bot1 = ResBlock(base * 4, base * 4, time_dim)
        self.bot2 = ResBlock(base * 4, base * 4, time_dim)
        # 上采样路径（注意 concat 后通道翻倍）
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1)  # 7→14
        self.dec2_1 = ResBlock(base * 4, base * 2, time_dim)
        self.dec2_2 = ResBlock(base * 2, base * 2, time_dim)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)      # 14→28
        self.dec1_1 = ResBlock(base * 2, base, time_dim)
        self.dec1_2 = ResBlock(base, base, time_dim)
        self.out = nn.Conv2d(base, in_ch, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        # 编码
        x1 = self.enc1_2(self.enc1_1(x, t_emb), t_emb)
        x2 = self.enc2_2(self.enc2_1(self.down1(x1), t_emb), t_emb)
        x3 = self.enc3_2(self.enc3_1(self.down2(x2), t_emb), t_emb)
        # bottleneck
        x3 = self.attn(x3)
        x3 = self.bot2(self.bot1(x3, t_emb), t_emb)
        # 解码（skip concat）
        x = self.up2(x3)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2_2(self.dec2_1(x, t_emb), t_emb)
        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1_2(self.dec1_1(x, t_emb), t_emb)
        return self.out(x)

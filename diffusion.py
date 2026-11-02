"""
diffusion.py — 加噪(q_sample) + 采样(sample)

核心公式：
    x_t = sqrt(ᾱ_t) · x0 + sqrt(1-ᾱ_t) · ε,   ε ~ N(0, I)

💡 单独测试本文件：
    python diffusion.py
    会验证 t=0 时 x_t≈原图、t=T-1 时≈纯噪声，并画出一张图的逐步加噪过程。
"""
import torch

from config import T, DEVICE
from schedule import make_schedule

def q_sample(x0, t, noise, sched=None):
    """
    给定原图 x0 和步数 t，一步加噪得到 x_t（t最大999，每个batch128个，每个样本t不同，循环不可行）

    x0, noise : (B, C, H, W)
    t         : (B,) 整数，每个样本可以不同步数
    返回      : x_t
    """
    if sched is None:
        sched = make_schedule()
    # gather 按样本索引取各自 t 对应的常数；reshape 成 (B,1,1,1) 才能和图片广播
    sqrt_ab = sched["sqrt_alpha_bars"].gather(0, t).view(-1, 1, 1, 1)
    sqrt_1mab = sched["sqrt_one_minus_alpha_bars"].gather(0, t).view(-1, 1, 1, 1)
    return sqrt_ab * x0 + sqrt_1mab * noise


def test_q_sample():
    """自检：t=0 时 x_t≈x0；t=T-1 时 x_t≈纯噪声。"""
    sched = make_schedule()
    x0 = torch.ones(2, 1, 4, 4)
    noise = torch.randn(2, 1, 4, 4)
    xt0 = q_sample(x0, torch.tensor([0, 0]), noise, sched)
    xtT = q_sample(x0, torch.tensor([T - 1, T - 1]), noise, sched)
    assert torch.allclose(xt0, x0, atol=1e-5), "t=0 应几乎等于原图"
    print("[diffusion] t=0 → x_t≈x0 ✅；t=T-1 → x_t≈噪声 ✅")


def plot_noising(x0, sched=None, save_path="noising.png", steps=(0, 250, 500, 750, 999)):
    """画图：同一张图在不同 t 下的加噪过程（保存 noising.png）。

    ❖ 这张图反映什么：
       直观验证"加噪公式实现正确"。
       t=0 应几乎就是原图，随 t 增大逐渐变成纯高斯噪声。
       如果 t=0 就已经糊了，说明 sqrt_alpha_bars 的 gather/reshape 错了
    """
    import matplotlib.pyplot as plt
    if sched is None:
        sched = make_schedule()
    imgs = []
    for t in steps:
        with torch.no_grad():
            xt = q_sample(x0, torch.tensor([t]), torch.randn_like(x0), sched)
        imgs.append((xt[0, 0].clamp(-1, 1) * 0.5 + 0.5).cpu().numpy())
    fig, axs = plt.subplots(1, len(steps), figsize=(len(steps) * 2, 2))
    for ax, (t, im) in zip(axs, zip(steps, imgs)):
        ax.imshow(im, cmap="gray")
        ax.set_title(f"t={t}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[diffusion] 已保存加噪过程图 -> {save_path}")


@torch.no_grad()
def sample(model, sched=None, shape=(1, 1, 28, 28), steps_show=5, device=DEVICE):
    """反向采样：从纯噪声 x_T 逐步去噪到 x_0。

    返回 (x, traj)：
      x    : 最终生成图 (B,C,H,W)，值域 [-1,1]
      traj : 中间若干帧，用于画"去噪演化"图
    """
    if sched is None:
        sched = make_schedule()
    x = torch.randn(shape, device=device)
    traj = []
    interval = max(1, T // steps_show)
    for t in reversed(range(T)):
        t_batch = torch.full((shape[0],), t, device=device)
        pred = model(x, t_batch)
        beta = sched["betas"][t]
        sac = sched["sqrt_recip_alphas"][t]
        so = sched["sqrt_one_minus_alpha_bars"][t]
        z = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        x = sac * (x - (1 - beta) / so * pred) + torch.sqrt(sched["posterior_variance"][t]) * z
        if t % interval == 0:
            traj.append(x.clone())
    return x, traj


def plot_sampling_trajectory(traj, save_path="sampling_traj.png"):
    """画图：展示 x_T → x_0 的去噪演化（保存 sampling_traj.png）。

    ❖ 这张图反映什么：
       反向过程是否真的在"逐步成型"。正确训练后，早期帧是模糊噪声，
       后期帧逐渐浮现数字轮廓。若全程都是灰噪，说明模型没训好，或
       time embedding 没接对（网络看不到 t，就学不出去噪）。
    """
    import matplotlib.pyplot as plt
    n = len(traj)
    fig, axs = plt.subplots(1, n, figsize=(n * 2, 2))
    for ax, im in zip(axs, traj):
        ax.imshow((im[0, 0].clamp(-1, 1) * 0.5 + 0.5).cpu().numpy(), cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[diffusion] 已保存采样轨迹图 -> {save_path}")


if __name__ == "__main__":
    test_q_sample()
    # 使用一张随机图演示加噪过程（真实训练时用数据集里的图更有意义）
    demo = torch.randn(1, 1, 28, 28)
    plot_noising(demo)

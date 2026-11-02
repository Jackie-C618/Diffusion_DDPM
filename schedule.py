"""
schedule.py — 前向过程加噪

DDPM 的前向过程是逐步往图片加高斯噪声。论文给出闭式一步加噪公式：
    x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps,  eps ~ N(0, I)

这里我们把所有和步数 t 相关的常数一次性算好缓存成 (T,) 的张量，
后面"加噪"(q_sample)和"采样"(sample)都直接按索引查表，不用重算。

💡 提供单独测试：
    python schedule.py
    会打印 α_bar 的首尾值做数值自检，并画出噪声 schedule 曲线图
"""
import torch
from config import T, BETA_START, BETA_END, DEVICE

def make_schedule(T=T, beta_start=BETA_START, beta_end=BETA_END, device=DEVICE):
    """
    计算噪声 schedule，返回 dict，所有值形状都是 (T,)。

    对应论文符号：
      betas                   : β_t                  
      alphas                  : α_t = 1 - β_t
      alpha_bars              : ᾱ_t = α_1·α_2·…·α_t  
      sqrt_alpha_bars         : √ᾱ_t
      sqrt_one_minus_alpha_bars : √(1 - ᾱ_t)         
      sqrt_recip_alphas        : 1/√α_t           
      posterior_variance       : 后验方差（采样用）
    """
    betas = torch.linspace(beta_start, beta_end, T, device=device)  # 等间距
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)            # 累乘，在 dim = 0 维度进行。得到ᾱ_t
    sqrt_alpha_bars = torch.sqrt(alpha_bars)
    sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)
    sqrt_recip_alphas = 1.0 / torch.sqrt(alphas)
    posterior_variance = betas                          # 论文中用的近似，未使用 ~betas

    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
        "sqrt_alpha_bars": sqrt_alpha_bars,
        "sqrt_one_minus_alpha_bars": sqrt_one_minus_alpha_bars,
        "sqrt_recip_alphas": sqrt_recip_alphas,
        "posterior_variance": posterior_variance,
    }


def test_schedule():
    """自检：确认 α_bar 从 ~1 衰减到 ~0（噪声逐步接管）。"""
    s = make_schedule()
    print("[schedule] α_bar 前5步 :", [round(v, 4) for v in s["alpha_bars"][:5].tolist()])
    print("[schedule] α_bar 末5步 :", [round(v, 4) for v in s["alpha_bars"][-5:].tolist()])
    assert s["alpha_bars"][0].item() > 0.99, "t=0 时应几乎无噪声（α_bar≈1）"
    assert s["alpha_bars"][-1].item() < 0.1, "t=T 时信号应几乎消失（α_bar≈0）"
    print("[schedule] 自检通过：α_bar 从 ~1 平滑降到 ~0 ✅")


def plot_schedule(sched, save_path="schedule.png"):
    """画图：噪声 schedule 曲线（保存到 schedule.png）。

    ❖ 这张图反映什么：
       · β_t（红）：每一步加多少噪声，线性从 1e-4 升到 0.02
       · ᾱ_t（绿）：累积保留的信号比例，从 1 平滑降到 0
       · √(1-ᾱ_t)（蓝）：噪声权重，从 0 升到 1
       它直观告诉你"信号随时间线性消失、噪声逐渐接管"，也是你换
       cosine schedule 时对比曲线形状是否更平滑的依据。
    """
    import matplotlib.pyplot as plt
    t = torch.arange(len(sched["betas"])).cpu().numpy()
    plt.figure(figsize=(7, 4))
    plt.plot(t, sched["betas"].cpu().numpy(), label=r"$\beta_t$ (每步加噪量)")
    plt.plot(t, sched["alpha_bars"].cpu().numpy(), label=r"$\bar\alpha_t$ (剩余信号)")
    plt.plot(t, sched["sqrt_one_minus_alpha_bars"].cpu().numpy(),
             label=r"$\sqrt{1-\bar\alpha_t}$ (噪声权重)")
    plt.xlabel("timestep t")
    plt.ylabel("value")
    plt.title("Noise Schedule (linear)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[schedule] 已保存曲线图 -> {save_path}")


if __name__ == "__main__":
    test_schedule()
    plot_schedule(make_schedule())

import torch


def make_schedule(T=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
    """第1步：噪声 schedule。
    把 DDPM 前向过程需要的全部常数一次性算好，存成 (T,) 的 tensor。
    """
    betas = torch.linspace(beta_start, beta_end, T, device=device)  # 线性 schedule
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)                      # ᾱ_t = ∏ α_i
    sqrt_alpha_bars = torch.sqrt(alpha_bars)
    sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)
    sqrt_recip_alphas = 1.0 / torch.sqrt(alphas)                   # 1/√α_t，采样用

    # 后验方差 β̃_t（采样用），clamp 防负数
    alpha_bars_prev = torch.cat([torch.ones(1, device=device), alpha_bars[:-1]])
    posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
    posterior_variance = posterior_variance.clamp(min=1e-20)

    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
        "sqrt_alpha_bars": sqrt_alpha_bars,
        "sqrt_one_minus_alpha_bars": sqrt_one_minus_alpha_bars,
        "sqrt_recip_alphas": sqrt_recip_alphas,
        "posterior_variance": posterior_variance,
    }

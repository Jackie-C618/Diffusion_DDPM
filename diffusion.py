import torch


class Diffusion:
    """第2步（q_sample 加噪）与第6步（sample 反向生成）。"""

    def __init__(self, schedule, device="cpu"):
        self.device = device
        for k, v in schedule.items():
            setattr(self, k, v)

    def q_sample(self, x0, t, noise=None):
        """闭式一步加噪：x_t = √ᾱ_t·x0 + √(1-ᾱ_t)·ε。
        x0: (B,1,28,28) 在 [-1,1]；t: (B,) 整数。
        """
        if noise is None:
            noise = torch.randn_like(x0)
        # gather 出对应 t 的常数，并 reshape 成 (B,1,1,1) 才能和图片广播
        sa = self.sqrt_alpha_bars[t].reshape(-1, 1, 1, 1)
        so = self.sqrt_one_minus_alpha_bars[t].reshape(-1, 1, 1, 1)
        return sa * x0 + so * noise

    @torch.no_grad()
    def sample(self, model, image_shape=(1, 28, 28), n=16):
        """从 x_T ~ N(0,I) 一步步去噪到 x_0。"""
        model.eval()
        x = torch.randn(n, *image_shape, device=self.device)
        steps = self.betas.shape[0]
        for t in reversed(range(steps)):
            t_batch = torch.full((n,), t, device=self.device, dtype=torch.long)
            pred = model(x, t_batch)
            sqrt_recip = self.sqrt_recip_alphas[t]
            so = self.sqrt_one_minus_alpha_bars[t]
            beta = self.betas[t]
            x = sqrt_recip * (x - (beta / so) * pred)     # 公式主体
            if t > 0:                                     # 最后一步不再加噪声
                x = x + torch.sqrt(self.posterior_variance[t]) * torch.randn_like(x)
        x = (x.clamp(-1, 1) + 1) / 2                      # 反归一化回 [0,1]
        return x

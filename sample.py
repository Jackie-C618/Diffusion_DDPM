"""
sample.py — 加载训练好的模型生成图片

💡 可单独测试本文件：
    python sample.py
    需要先用 train.py 训出 ckpt_last.pt。会生成两张图：
      · samples.png        4×4 生成数字网格（最终成果）
      · sampling_traj.png  去噪演化过程（见 diffusion.plot_sampling_trajectory）
"""
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CKPT, SAMPLES_PNG, DEVICE, SEED
from model import UNet
from diffusion import sample, plot_sampling_trajectory


def save_grid(x, n=4, save_path=SAMPLES_PNG):
    """画图：n×n 生成数字网格（保存 samples.png）。

    ❖ 这张图反映什么：
       最终成果。正确训练后应是清晰可辨的手写数字，是判断
       "你的 DDPM 实现到底对不对"最直观的证据。
    """
    x = (x.clamp(-1, 1) * 0.5 + 0.5).cpu()          # [-1,1] -> [0,1]
    fig, axs = plt.subplots(n, n, figsize=(n, n))
    for i, ax in enumerate(axs.flat):
        ax.imshow(x[i, 0], cmap="gray")
        ax.axis("off")
    plt.tight_layout()    # 自动调节边距
    plt.savefig(save_path)
    plt.close()
    print(f"[sample] 已保存生成图网格 -> {save_path}")


def generate(n=16):
    torch.manual_seed(SEED)
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.eval()
    x, traj = sample(model, shape=(n, 1, 28, 28), steps_show=6)
    save_grid(x, n=int(n ** 0.5))
    plot_sampling_trajectory(traj)
    return x


if __name__ == "__main__":
    generate()

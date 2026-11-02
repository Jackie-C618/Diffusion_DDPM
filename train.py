"""
train.py — 训练直接

每个 batch 做的事（核心就这几行）：
    1) x0   = 一个 batch 的原图
    2) t    = 随机步数 (B,)
    3) noise= 随机噪声
    4) xt   = q_sample(x0, t, noise)        # 加噪
    5) pred = model(xt, t)                  # 预测噪声
    6) loss = MSE(pred, noise)              # 和真噪声比
    7) 反向传播 + 更新参数

💡 单独测试本文件：
    python train.py
    会先跑一次 model 自检，再开始训练；每个 epoch 保存权重和 loss 曲线图。
    先改 config.EPOCHS=1 跑通看效果，再调大。
"""
import torch
from torch.optim import Adam
from tqdm import tqdm  # 进度条
import matplotlib
matplotlib.use("Agg")  # 无弹窗，只往文件里写
import matplotlib.pyplot as plt

from config import T, LR, EPOCHS, CKPT, DEVICE, SEED
from dataset import get_mnist_dataloader
from model import UNet, test_model
from diffusion import q_sample
from schedule import make_schedule


def save_loss_curve(losses, save_path="loss_curve.png"):
    """画图：训练 loss 曲线（保存 loss_curve.png）。

    ❖ 这张图反映什么：
       模型有没有在学。MNIST 上应平稳下降、最终收敛到较低值。
       若 loss 不降/爆炸，先回查 schedule 与 q_sample 的数值，
       再看 LR 是否过大（常见诱因）。
    """
    plt.figure(figsize=(7, 4))
    plt.plot(losses)
    plt.xlabel("step")
    plt.ylabel("MSE loss")
    plt.title("Training Loss")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train():
    torch.manual_seed(SEED)
    sched = make_schedule()
    loader = get_mnist_dataloader()
    model = UNet().to(DEVICE)
    opt = Adam(model.parameters(), lr=LR)

    losses = []
    step = 0
    for epoch in range(EPOCHS):
        for x0, _ in tqdm(loader, desc=f"epoch {epoch+1}/{EPOCHS}"): # 进度条
            x0 = x0.to(DEVICE)                        
            B = x0.shape[0]
            t = torch.randint(0, T, (B,), device=DEVICE)
            noise = torch.randn_like(x0)
            xt = q_sample(x0, t, noise, sched)        # 加噪
            pred = model(xt, t)                       # 预测噪声
            loss = torch.nn.functional.mse_loss(pred, noise)

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())
            step += 1
            if step % 200 == 0:
                print(f"  step {step}  loss {loss.item():.4f}")

        torch.save(model.state_dict(), CKPT)
        save_loss_curve(losses)
        print(f"[train] epoch {epoch+1} 完成，已保存 {CKPT} 与 loss_curve.png")

    return model


if __name__ == "__main__":   
    test_model()
    train()

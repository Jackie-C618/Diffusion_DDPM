import torch
from schedule import make_schedule
from model import UNet
from diffusion import Diffusion
from dataset import get_mnist

# ---- 设备：有 GPU 用 GPU，没有自动用 CPU ----
device = "cuda" if torch.cuda.is_available() else "cpu"
T = 1000

# ---- 组装各部分 ----
schedule = make_schedule(T=T, device=device)
model = UNet(in_ch=1, base=64, time_dim=256).to(device)
diff = Diffusion(schedule, device=device)
loader = get_mnist(batch_size=128)
opt = torch.optim.Adam(model.parameters(), lr=2e-4)
loss_fn = torch.nn.MSELoss()

# ---- 训练循环（第5步）----
model.train()
for epoch in range(50):
    for x0, _ in loader:
        x0 = x0.to(device)
        B = x0.shape[0]
        t = torch.randint(0, T, (B,), device=device)   # 每个样本随机一个时间步
        noise = torch.randn_like(x0)                    # 真实噪声 ε
        xt = diff.q_sample(x0, t, noise)                # 加噪得到 x_t
        pred = model(xt, t)                             # 网络预测噪声 ε̂
        loss = loss_fn(pred, noise)                     # 预测噪声 vs 真实噪声 的 MSE

        opt.zero_grad()     # 清空上一步梯度
        loss.backward()     # 反向传播算梯度
        opt.step()          # 更新参数

    print(f"epoch {epoch:02d}  loss {loss.item():.4f}")
    torch.save(model.state_dict(), "ckpt_last.pt")     # 每个 epoch 覆盖保存
print("训练完成，已保存 ckpt_last.pt")

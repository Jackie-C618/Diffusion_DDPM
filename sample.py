import torch
from torchvision.utils import make_grid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from schedule import make_schedule
from model import UNet
from diffusion import Diffusion

device = "cuda" if torch.cuda.is_available() else "cpu"
schedule = make_schedule(T=1000, device=device)
model = UNet(in_ch=1, base=64, time_dim=256).to(device)
model.load_state_dict(torch.load("ckpt_last.pt", map_location=device))
diff = Diffusion(schedule, device=device)

imgs = diff.sample(model, n=16)          # 生成 16 张 (16,1,28,28)，范围 [0,1]
grid = make_grid(imgs, nrow=4, padding=2)
plt.figure(figsize=(4, 4))
plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
plt.axis("off")
plt.savefig("samples.png", dpi=150, bbox_inches="tight")
print("已保存 samples.png")

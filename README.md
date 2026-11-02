# 🎯 DDPM on MNIST

> 用 **PyTorch** 实现的基于 U-net 的 DDPM（Denoising Diffusion Probabilistic Models），在手写数字数据集 **MNIST** 上训练。

本项目作为**学习扩散模型的练习**

每个文件均可单独运行

---

## ✨ 特性

- 🧩 **模块化 U-Net**：`TimeEmbedding` / `ResidualBlock` / `DownBlock` / `UpBlock` / `Attention` 各个功能拆分成函数
- 🧪 **分步可测**：每个文件底部都有 `if __name__ == "__main__":` 自检，`python xxx.py` 单独验证
- 📊 **可视化图**，直观反映噪声调度、加噪过程、训练损失、去噪演化、最终生成
- 🎛️ **超参数集中**在 `config.py`，改模型大小/步数只动一处

---

## 📂 目录结构

```
ddpm_mnist/
├── config.py      ⚙️ 所有超参数（T、学习率、通道数…）集中管理
├── schedule.py    📈 第1步：噪声 schedule（β / α / ᾱ 等常数）+ 噪声曲线图
├── model.py       🧠 第3步：U-Net 全套零件 + 形状自检
├── diffusion.py   🔀 第2步加噪 q_sample + 第6步采样 sample + 两张图
├── dataset.py     🗂️ 第0步：MNIST 加载，归一化到 [-1,1]
├── train.py       🚀 第5步：训练循环 + loss 曲线图
├── sample.py      🖼️ 第6步：加载模型生成图片 + 去噪轨迹图
└── .gitignore     🚫 忽略数据集 / 权重 / 缓存
```

---

## ⚙️ 安装

```bash
pip install torch torchvision matplotlib tqdm
```

> 💡 没有 GPU 也能跑（MNIST 很小，CPU只是慢一点）。想提速就把 `config.py` 里的 `DEVICE` 自动检测为 `cuda` 即可。

---

## 🧪 分步测试（强烈建议先跑）

不用一口气跑通全部，每一步都能单独验证：

```bash
python schedule.py     # ✅ 打印 α_bar 首尾值 + 出 schedule.png
python model.py        # ✅ 随机张量跑 forward，断言形状/梯度
python diffusion.py    # ✅ 验证 t=0≈原图、t=T-1≈噪声 + 出 noising.png
python dataset.py      # ✅ 取一个 batch 验证形状/值域（首次会下载 MNIST）
```

---

## 🚀 训练与采样

```bash
# 1) 训练
python train.py
#   → 每个 epoch 保存 ckpt_last.pt 和 loss_curve.png

# 2) 用训好的模型生成图片
python sample.py
#   → 生成 samples.png（4×4 数字网格）和 sampling_traj.png（去噪过程）
```

---

## 📊 图文件

| 图文件 | 来源 | 它反映什么 |
|---|---|---|
| `schedule.png` | `schedule.py` | 噪声调度曲线：β 线性上升、ᾱ 从 1 降到 0、噪声权重从 0 升到 1 —— 确认"信号随时间消失、噪声接管" |
| `noising.png` | `diffusion.py` | 同一张图在 t=0/250/500/750/999 的加噪过程 —— **验证加噪公式写对没**，t=0 必须≈原图 |
| `loss_curve.png` | `train.py` | 训练 loss 曲线 —— 平稳下降=在学；不降/爆炸=先查 schedule 与 q_sample 数值 |
| `sampling_traj.png` | `sample.py` | x_T → x_0 的去噪演化 —— 正确训练后早期是糊噪、后期浮现数字轮廓 |
| `samples.png` | `sample.py` | 最终 4×4 数字网格 —— 你实现到底对不对的"判决书" |

---

## 🧮 核心公式

**前向加噪**：

```
x_t = √ᾱ_t · x₀ + √(1 - ᾱ_t) · ε,   ε ~ N(0, I)
```

**训练目标**（预测噪声）：

```
loss = MSE( ε ,  ε_θ(x_t, t) )
```

**反向采样**（从纯噪声逐步去噪到 x₀）：按 t 从 T-1 到 0 迭代 DDPM 采样公式。

---

## 🎛️ 可调超参数（`config.py`）

| 参数 | 默认 | 说明 |
|---|---|---|
| `T` | 1000 | 扩散总步数 |
| `BETA_START / END` | 1e-4 / 0.02 | 线性噪声 schedule 起止 |
| `BASE_CHANNELS` | 64 | U-Net 基础通道数 |
| `DEPTH` | 2 | 下采样次数（28×28 下采样 2 次形状能完美往返） |
| `LR` | 2e-4 | 学习率 |
| `BATCH_SIZE` | 128 | 批大小 |

---

## ⭐ 排错清单

- **形状对不上** → 90% 是 `gather` 后 `reshape` 漏了 `(B,1,1,1)`，或上/下采样通道没对齐。
- **loss 是 NaN** → 查归一化值域、`LR`、`posterior_variance` 是否出负数。
- **生成全是灰/噪** → 训练不够，或时间嵌入没接好（网络看不到 t 就学不出去噪）。
- **显存爆** → 减小 `BATCH_SIZE` / `BASE_CHANNELS`，MNIST 先别纠结 GPU。

---

## 📚 参考

- Ho et al., *Denoising Diffusion Probabilistic Models* (2020)


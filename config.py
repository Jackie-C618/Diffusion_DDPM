"""
config.py —— 超参数
"""
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 数据集 ----------
BATCH_SIZE = 128    # 一次训练选取的样本数
IMAGE_SIZE = 28     # 28*28
CHANNELS = 1        # MNIST 是单通道灰度图
# [0,1] -> 映射到 [-1,1]
NORM_MEAN = 0.5
NORM_STD = 0.5      

# ---------- 扩散过程 ----------
T = 1000            # 扩散总步数
BETA_START = 1e-4   # 线性 schedule 起点（每步加噪量下限）
BETA_END = 0.02     # 线性 schedule 终点（每步加噪量上限）

# ---------- 模型 ----------
BASE_CHANNELS = 64
# 下采样次数。28 是偶数，下采样 2 次正好 28->14->7，上采样原路返回最稳。
# 想更深就改成 3（会用到 ConvTranspose2d 的对齐，模型已处理）。
DEPTH = 2
ATTN_AT = 1         # >=0 表示在 bottleneck 加自注意力；-1 表示不加
TIME_EMB_DIM = 256  # 时间嵌入维度

# ---------- 训练 ----------
LR = 2e-4
EPOCHS = 20
SEED = 42    # 随机数种子

# ---------- 路径 ----------
DATA_DIR = "data"
CKPT = "ckpt_last.pt"         # 权重
SAMPLES_PNG = "samples.png"

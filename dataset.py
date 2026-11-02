"""
dataset.py — 预加载 MNIST，并把像素归一化到 [-1,1]。

DDPM 论文里默认 x0 是 [-1,1] 上的连续量，所以这里用
ToTensor([0,1]) → Normalize(0.5,0.5) 把它映射到 [-1,1]。
（其实也可在此区间内处理，但需保持前后一致，也即加噪/sample 都按这个区间来）

💡 可单独测试本文件：
    python dataset.py
    会取一个 batch，确认形状是 (B,1,28,28)、像素在 [-1,1]
"""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from config import BATCH_SIZE, DATA_DIR, NORM_MEAN, NORM_STD, IMAGE_SIZE, CHANNELS

# 把 PIL 图变成 [-1,1] 的张量
def get_transform():

    return transforms.Compose([
        transforms.ToTensor(),                          # 归一化[0,1]
        # 由于Normalize两个变量需要为tuple类型（多通道处理），因此写为 (0.5,0)
        transforms.Normalize((NORM_MEAN,), (NORM_STD,)),  # Normalize 执行 (x-mean)/std -> [-1,1]
    ])


def get_mnist_dataloader(train=True):
    ds = datasets.MNIST(
        root=DATA_DIR, train=train, download=True, transform=get_transform()
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=train, num_workers=0) # num_workers: DataLoader 加载数据的子进程数，一般大数据集设置大于零

# 自检：取一个 batch，确认形状与值域
def test_dataset():

    loader = get_mnist_dataloader()
    x, y = next(iter(loader))
    print("[dataset] batch 形状:", tuple(x.shape), "标签形状:", tuple(y.shape))
    print("[dataset] 像素范围: min=%.3f max=%.3f" % (x.min(), x.max()))
    # 由于Normalize只做线性变换不做裁剪，因此assert的检查是必要的，检查输入是否越界
    assert x.shape[1:] == (CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    assert x.min() >= -1.01 and x.max() <= 1.01, "像素应在 [-1,1]" # 断言失败时的报错信息
    print("[dataset] 自检通过 ✅")

# 只有直接运行本文件时才执行以下语句
if __name__ == "__main__":
    test_dataset()

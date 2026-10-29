from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_mnist(batch_size=64, root="./data"):
    """第0步：加载 MNIST，并把像素从 [0,1] 归一化到 [-1,1]。"""
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train = datasets.MNIST(root=root, train=True, download=True, transform=tf)
    return DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)

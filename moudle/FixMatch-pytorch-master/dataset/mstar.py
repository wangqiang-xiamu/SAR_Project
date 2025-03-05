import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class MSTARDataset(Dataset):
    def __init__(self, root, transform=None):
        """
        初始化MSTAR数据集。

        参数:
            root (str): 数据集根目录。
            transform (callable, optional): 数据预处理函数。
        """
        self.root = root
        self.transform = transform
        self.classes = os.listdir(root)  # 获取所有类别文件夹
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}  # 类别到索引的映射
        self.images = self._load_images()  # 加载所有图像路径和标签

    def _load_images(self):
        """
        加载所有图像路径和对应的标签。

        返回:
            list: 包含 (图像路径, 标签) 的列表。
        """
        images = []
        for cls in self.classes:
            cls_dir = os.path.join(self.root, cls)  # 类别文件夹路径
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)  # 图像路径
                images.append((img_path, self.class_to_idx[cls]))  # 添加 (路径, 标签)
        return images

    def __len__(self):
        """
        返回数据集的大小。

        返回:
            int: 数据集中的样本数量。
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        获取单个样本。

        参数:
            idx (int): 样本索引。

        返回:
            tuple: (图像, 标签)。
        """
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('L')  # 加载图像并转换为灰度
        if self.transform:
            image = self.transform(image)  # 数据预处理
        return image, label


def get_mstar(args, root):
    """
    获取MSTAR数据集的训练和测试数据。

    参数:
        args: 命令行参数。
        root (str): 数据集根目录。

    返回:
        tuple: (labeled_dataset, unlabeled_dataset, test_dataset)。
    """
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为张量
    ])
    labeled_dataset = MSTARDataset(root=os.path.join(root, 'labeled'), transform=transform)
    unlabeled_dataset = MSTARDataset(root=os.path.join(root, 'unlabeled'), transform=transform)
    test_dataset = MSTARDataset(root=os.path.join(root, 'test'), transform=transform)
    return labeled_dataset, unlabeled_dataset, test_dataset
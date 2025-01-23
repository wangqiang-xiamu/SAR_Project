import os
from PIL import Image
from torch.utils.data import Dataset
import random


class SARDataset(Dataset):
    def __init__(self, img_dir, class_names, transform=None, is_unlabeled=False, max_size=100):
        """
        初始化数据集类
        :param img_dir: 数据集所在的根目录
        :param class_names: 类别名称列表（例如 ["2S1", "BMP2", "T62", ...]）
        :param transform: 图像变换函数（例如，数据增强，归一化等）
        :param is_unlabeled: 如果是无标签数据集，设为True
        :param max_size: 控制数据集大小，None 表示没有限制，指定值表示最大数量
        """
        self.img_dir = img_dir
        self.class_names = class_names
        self.transform = transform
        self.is_unlabeled = is_unlabeled
        self.max_size = max_size

        # 存储图像路径和标签
        self.img_paths = []
        self.labels = []

        # 创建一个字典来存储每个类别的图像路径
        self.class_images = {class_name: [] for class_name in self.class_names}

        # 遍历类别文件夹，加载所有图像路径
        for label, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.img_dir, class_name)
            if os.path.isdir(class_dir):
                for filename in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, filename)
                    if img_path.lower().endswith((".jpg", ".png", ".jpeg")):
                        self.class_images[class_name].append(img_path)

        # 计算每个类别应加载的图像数量
        total_classes = len(self.class_images)  # 类别总数
        images_per_class = self.max_size // total_classes  # 每个类别分配的图像数量

        # 确保每个类别加载的图像数量为 10 或者该类别图像数量
        for class_name, img_paths in self.class_images.items():
            sampled_paths = random.sample(img_paths, min(images_per_class, len(img_paths)))  # 采样每个类别的图像
            self.img_paths.extend(sampled_paths)  # 将采样后的图像路径添加到 img_paths 中
            self.labels.extend([self.class_names.index(class_name)] * len(sampled_paths))  # 添加对应的标签

        if len(self.img_paths) == 0:
            raise ValueError(f"Dataset is empty. No images found in {self.img_dir}")

        print(f"Found {len(self.img_paths)} images in {self.img_dir} with balanced classes.")

    def __len__(self):
        """
        返回数据集的大小（限制为 max_size）
        """
        return len(self.img_paths)

    def __getitem__(self, idx):
        """
        获取指定索引的图像和标签（如果是有标签数据集）
        :param idx: 索引
        :return: 图像和标签（或仅图像，如果是无标签数据集）
        """
        img_path = self.img_paths[idx]  # 获取指定索引的图像路径
        image = Image.open(img_path)  # 使用 PIL 打开图像
        if self.transform:
            image = self.transform(image)  # 如果指定了转换函数，则应用转换

        if self.is_unlabeled:
            return image  # 如果是无标签数据集，仅返回图像
        else:
            label = self.labels[idx]  # 获取对应的标签
            return image, label  # 返回图像和标签

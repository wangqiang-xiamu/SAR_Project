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
        self.img_dir = img_dir  # 数据集所在的根目录
        self.class_names = class_names  # 类别名称列表
        self.transform = transform  # 图像变换函数（用于数据预处理或增强）
        self.is_unlabeled = is_unlabeled  # 如果是无标签数据集，则为 True
        self.max_size = max_size  # 数据集的最大大小

        # 存储图像路径和标签的列表
        self.img_paths = []  # 存放所有图像的路径
        self.labels = []  # 存放所有图像的标签

        # 创建一个字典，用来存储每个类别的图像路径
        self.class_images = {class_name: [] for class_name in self.class_names}

        # 遍历每个类别的文件夹，加载图像路径
        for label, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.img_dir, class_name)  # 获取类别文件夹路径
            if os.path.isdir(class_dir):  # 确保类别文件夹存在
                for filename in os.listdir(class_dir):  # 遍历文件夹中的每个文件
                    img_path = os.path.join(class_dir, filename)  # 获取每个图像文件的完整路径
                    if img_path.lower().endswith((".jpg", ".png", ".jpeg")):  # 检查文件扩展名
                        self.class_images[class_name].append(img_path)  # 将图像路径添加到对应类别的列表

        # 计算每个类别应加载的图像数量
        total_classes = len(self.class_images)  # 获取类别总数
        images_per_class = self.max_size // total_classes  # 计算每个类别分配的图像数量

        # 确保每个类别加载的图像数量是合理的（不能超过该类别的图像数量）
        for class_name, img_paths in self.class_images.items():
            # 对每个类别进行随机采样，确保采样数量不超过该类别的实际图像数量
            sampled_paths = random.sample(img_paths, min(images_per_class, len(img_paths)))
            self.img_paths.extend(sampled_paths)  # 将采样后的图像路径添加到总列表
            self.labels.extend([self.class_names.index(class_name)] * len(sampled_paths))  # 为每个采样的图像添加标签

        # 如果最终没有图像，抛出异常
        if len(self.img_paths) == 0:
            raise ValueError(f"Dataset is empty. No images found in {self.img_dir}")

        # 打印数据集的大小和类别均衡情况
        print(f"Found {len(self.img_paths)} images in {self.img_dir} with balanced classes.")

    def __len__(self):
        """
        返回数据集的大小（限制为 max_size）
        """
        return len(self.img_paths)  # 返回图像路径列表的长度，表示数据集的大小

    def __getitem__(self, idx):
        """
        获取指定索引的图像和标签（如果是有标签数据集）
        :param idx: 索引
        :return: 图像和标签（或仅图像，如果是无标签数据集）
        """
        img_path = self.img_paths[idx]  # 根据索引获取图像路径
        image = Image.open(img_path)  # 使用 PIL 打开图像
        if self.transform:
            image = self.transform(image)  # 如果指定了转换函数，则应用转换（如数据增强或归一化）

        if self.is_unlabeled:
            return image  # 如果是无标签数据集，仅返回图像
        else:
            label = self.labels[idx]  # 获取对应的标签
            return image, label  # 返回图像和标签。如果是无标签数据集只返回图像

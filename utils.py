import os
from PIL import Image
from torch.utils.data import Dataset

# 无标签数据集：通过设置 is_unlabeled=True，__getitem__ 方法只返回图像，不返回标签。
# 有标签数据集：__getitem__ 方法返回图像和标签，适用于训练。
# 路径和文件夹结构：
# 对于有标签数据集，图像应存放在 mstar-train 中的各个类文件夹下。
# 对于无标签数据集，图像应存放在 mstar-unlabeled 中的类文件夹下。

class SARDataset(Dataset):
    def __init__(self, img_dir, class_names, transform=None, is_unlabeled=False, max_size=None):
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

        # 加载图像数据
        for label, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.img_dir, class_name)
            if os.path.isdir(class_dir):
                for filename in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, filename)
                    if img_path.lower().endswith((".jpg", ".png", ".jpeg")):
                        self.img_paths.append(img_path)
                        if not self.is_unlabeled:
                            self.labels.append(label)
                    if self.max_size and len(self.img_paths) >= self.max_size:
                        break
            if self.max_size and len(self.img_paths) >= self.max_size:
                break

        if len(self.img_paths) == 0:
            raise ValueError(f"Dataset is empty. No images found in {self.img_dir}")

        print(f"Found {len(self.img_paths)} images in {self.img_dir}")

    def __len__(self):
        """
        返回数据集的大小（限制为 max_size）
        """
        return min(len(self.img_paths), self.max_size) if self.max_size else len(self.img_paths)

    def __getitem__(self, idx):
        """
        获取指定索引的图像和标签（如果是有标签数据集）
        :param idx: 索引
        :return: 图像和标签（或仅图像，如果是无标签数据集）
        """
        img_path = self.img_paths[idx]
        image = Image.open(img_path)  # 加载图片
        if self.transform:
            image = self.transform(image)

        if self.is_unlabeled:
            return image  # 无标签数据集只返回图像
        else:
            label = self.labels[idx]
            return image, label  # 有标签数据集返回图像和标签
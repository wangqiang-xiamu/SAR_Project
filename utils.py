import os
from PIL import Image
from torch.utils.data import Dataset

# 无标签数据集：通过设置 is_unlabeled=True，__getitem__ 方法只返回图像，不返回标签。
# 有标签数据集：__getitem__ 方法返回图像和标签，适用于训练。
# 路径和文件夹结构：
# 对于有标签数据集，图像应存放在 mstar-train 中的各个类文件夹下。
# 对于无标签数据集，图像应存放在 mstar-unlabeled 中的类文件夹下。

class SARDataset(Dataset):
    def __init__(self, img_dir, class_names, transform=None, is_unlabeled=False):
        """
        初始化数据集类
        :param img_dir: 数据集所在的根目录
        :param class_names: 类别名称列表（例如 ["2S1", "BMP2", "T62", ...]）
        :param transform: 图像变换函数（例如，数据增强，归一化等）
        :param is_unlabeled: 如果是无标签数据集，设为True
        """
        self.img_dir = img_dir
        self.class_names = class_names
        self.transform = transform
        self.is_unlabeled = is_unlabeled

        # 存储图像路径和标签
        self.img_paths = []
        self.labels = []

        print(f"Loading images from {self.img_dir}...")  # 调试信息
        for label, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.img_dir, class_name)
            print(f"Checking class directory: {class_dir}")  # 调试信息
            if os.path.isdir(class_dir):
                print(f"Found class folder: {class_dir}")  # 调试信息
                for filename in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, filename)
                    # 检查文件类型
                    if img_path.lower().endswith((".jpg", ".png", ".jpeg")):
                        self.img_paths.append(img_path)
                        if not self.is_unlabeled:
                            self.labels.append(label)  # 只有在有标签数据集时才添加标签
                        print(f"Added image: {img_path}")  # 调试信息
                    else:
                        print(f"Skipping non-image file: {img_path}")  # 非图像文件跳过
            else:
                print(f"Warning: {class_dir} is not a directory.")  # 如果类文件夹不存在，给出警告

        # 检查是否成功加载数据
        if len(self.img_paths) == 0:
            raise ValueError(f"Dataset is empty. No images found in {self.img_dir}")

        print(f"Found {len(self.img_paths)} images in {self.img_dir}")  # 输出加载到的图像数量

    def __len__(self):
        """
        返回数据集的大小
        """
        return len(self.img_paths)

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

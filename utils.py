import os
from PIL import Image
from torch.utils.data import Dataset


class SARDataset(Dataset):
    def __init__(self, img_dir, class_names, transform=None):
        self.img_dir = img_dir
        self.class_names = class_names
        self.transform = transform

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
                        self.labels.append(label)
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
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path)  # 加载图片
        if self.transform:
            image = self.transform(image)
        return image, label

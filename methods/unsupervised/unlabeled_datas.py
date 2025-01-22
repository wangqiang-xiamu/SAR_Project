import os
import random
from PIL import Image
import torchvision.transforms as transforms

# 数据增强
transform = transforms.Compose([
    transforms.RandomRotation(30),  # 随机旋转图像
    transforms.RandomResizedCrop(224),  # 随机裁剪并调整大小
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ColorJitter(brightness=0.5, contrast=0.5),  # 随机调整亮度与对比度
    transforms.ToTensor(),  # 转换为张量
])

def augment_data(data_path, output_dir, num_augmentations=3):
    """
    生成增强的数据（无标签数据）
    data_path: 有标签数据的路径
    output_dir: 生成的无标签数据存储路径
    num_augmentations: 每个图像生成的增强数量
    """
    categories = os.listdir(data_path)
    for category in categories:
        category_path = os.path.join(data_path, category)
        if os.path.isdir(category_path):
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                if img_name.endswith('.jpg') or img_name.endswith('.png') or img_name.endswith('.jpeg'):
                    image = Image.open(img_path)
                    for _ in range(num_augmentations):
                        # 应用数据增强
                        augmented_image_tensor = transform(image)

                        # 将增强后的 Tensor 转换为 PIL 图像
                        augmented_image = transforms.ToPILImage()(augmented_image_tensor)

                        # 构造保存路径
                        augmented_image_path = os.path.join(output_dir, category,
                                                            f"{img_name}_{random.randint(1000, 9999)}.jpg")
                        os.makedirs(os.path.dirname(augmented_image_path), exist_ok=True)

                        # 保存增强后的图像
                        augmented_image.save(augmented_image_path)


if __name__ == '__main__':
    # 生成无标签数据
    augment_data('/Users/xiamu/PycharmProjects/SAR_Project/data/MSTAR/mstar-train',
                 '/Users/xiamu/PycharmProjects/SAR_Project/data/MSTAR/mstar-unlabeled')

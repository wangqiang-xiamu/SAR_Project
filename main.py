import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights

# 数据增强示例：包括调整大小、转换为Tensor和归一化
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # 将灰度图转换为3通道RGB图像
    transforms.Resize((224, 224)),  # 调整图像大小为224x224
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用预训练模型的标准化
])

# 类别名称示例，和数据集的文件夹结构一致
class_names = ['2S1', 'BMP2', 'BRDM_2', 'BTR60', 'BTR70', 'D7', 'T62', 'T72', 'ZIL131', 'ZSU_23_4']


# 自定义数据集类
class SARDataset(Dataset):
    def __init__(self, img_dir, class_names, transform=None):
        self.img_dir = img_dir
        self.class_names = class_names
        self.transform = transform

        self.img_paths = []
        self.labels = []

        for label, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.img_dir, class_name)
            if os.path.isdir(class_dir):
                for filename in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, filename)
                    if img_path.endswith(('.jpg', '.png', '.jpeg')):
                        self.img_paths.append(img_path)
                        self.labels.append(label)

        if len(self.img_paths) == 0:
            raise ValueError(f"No images found in {img_dir}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label

def main():
    # MSTAR数据集的训练和测试集文件夹路径
    train_img_dir = './data/MSTAR/mstar-train'  # 训练数据集路径
    test_img_dir = './data/MSTAR/mstar-test'  # 测试数据集路径

    # 创建训练数据集实例
    train_dataset = SARDataset(img_dir=train_img_dir, class_names=class_names, transform=transform)

    # 创建测试数据集实例
    test_dataset = SARDataset(img_dir=test_img_dir, class_names=class_names, transform=transform)

    # 创建训练数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=6, pin_memory=True)

    # 创建测试数据加载器
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=6, pin_memory=True)

    # 加载预训练的ResNet18模型
    # 加载ResNet18模型，使用最新的预训练权重
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # 修改最后的全连接层，以适应你的数据集
    model.fc = nn.Linear(model.fc.in_features, len(class_names))  # 这里的 len(class_names) 是你的类别数

    # 将模型移动到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 分类任务常用的损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器

    # 训练过程
    num_epochs = 10  # 设置训练轮数
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}")
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_dataloader):
            print(f"Loading batch {i + 1}")
            images, labels = images.to(device), labels.to(device)

            # 清零优化器的梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(images)

            # 计算损失
            loss = criterion(outputs, labels)

            # 反向传播并更新权重
            loss.backward()
            optimizer.step()

            # 统计损失和准确度
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # 输出每一轮的损失和准确度
        epoch_loss = running_loss / len(train_dataloader)
        epoch_acc = correct / total * 100
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    # 保存训练好的模型
    torch.save(model.state_dict(), 'sar_model.pth')

    # 测试模型
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 输出测试准确度
    test_acc = correct / total * 100
    print(f"Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()

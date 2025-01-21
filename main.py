import os
from utils import SARDataset
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,models
import torch.nn as nn
import torch.optim as optim
from methods.mixup import mixup_data
from methods.fixmatch import fixmatch_criterion
# MixUp + FixMatch组合：在传统MixUp的基础上加入FixMatch，可以更好地利用未标记数据。
# MixUp通过增强训练数据的多样性，而FixMatch通过伪标签和一致性损失进一步加强模型的泛化能力。

# 数据增强示例：包括调整大小、转换为Tensor和归一化，将图像转化为神经网络模型可以处理的格式。
# ResNet18，预训练的模型（ResNet18）基于三通道的图像进行训练的。
# 将输入图像从灰度图（单通道）转换为三通道的RGB图像。
# 数据增强，包括调整大小、转换为Tensor和归一化
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # 将灰度图转换为3通道RGB图像
    transforms.Resize((224, 224)),  # 调整图像大小为224x224
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 预训练模型的标准化
])

class_names = ['2S1', 'BMP2', 'BRDM_2', 'BTR60', 'BTR70', 'D7', 'T62', 'T72', 'ZIL131', 'ZSU_23_4']

def main():
    #测试
    train_img_dir = './data/MSTAR/mstar-train'  # 有标签的训练数据
    test_img_dir = './data/MSTAR/mstar-test'  # 测试数据
    # train_img_dir = './data/MSTAR/mstar-train'  # 有标签的训练数据
    # test_img_dir = './data/MSTAR/mstar-test'  # 测试数据
    unlabeled_img_dir = './data/MSTAR/mstar-unlabeled'  # 无标签数据路径

    # 创建训练数据集和测试数据集
    train_dataset = SARDataset(img_dir=train_img_dir, class_names=class_names, transform=transform)
    test_dataset = SARDataset(img_dir=test_img_dir, class_names=class_names, transform=transform)

    # 创建无标签数据集
    unlabeled_dataset = SARDataset(img_dir=unlabeled_img_dir, class_names=class_names, transform=transform, is_unlabeled=True)

    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=6, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=6, pin_memory=True)
    unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=32, shuffle=True, num_workers=6, pin_memory=True)

    # 加载ResNet18模型ƒ
    # 使用新的 weights 参数加载模型
    # ResNet18_Weights.DEFAULT: 使用最新推荐的预训练权重。
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    model.fc = nn.Linear(model.fc.in_features, len(class_names))  # 修改输出层

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 有标签数据的损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

    # 训练过程
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for (images, labels) in train_dataloader:
            images, labels = images.to(device), labels.to(device)

            # 清零优化器的梯度
            optimizer.zero_grad()

            # MixUp数据增强
            mixed_images, mixed_labels, lam = mixup_data(images, labels, alpha=1.0)
            # 前向传播
            outputs = model(mixed_images)
            loss = criterion(outputs, mixed_labels)

            # 反向传播
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_dataloader)
        epoch_acc = correct / total * 100
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

        # 使用FixMatch进行无标签数据训练
        model.eval()
        with torch.no_grad():
            for unlabeled_images in unlabeled_dataloader:  # 直接获取图像
                unlabeled_images = unlabeled_images.to(device)

                # 为无标签数据生成伪标签
                weak_images = augment_data(unlabeled_images, weak=True)
                strong_images = augment_data(unlabeled_images, weak=False)

                # 获取模型的预测
                weak_logits = model(weak_images)  # 计算弱增强的预测
                strong_logits = model(strong_images)  # 计算强增强的预测

                weak_logits.requires_grad_()  # 确保 weak_logits 需要梯度
                strong_logits.requires_grad_()  # 确保 strong_logits 需要梯度

                # 计算伪标签
                weak_probs = F.softmax(weak_logits, dim=1)
                max_probs, pseudo_labels = weak_probs.max(dim=1)

                # 伪标签过滤
                mask = max_probs.ge(0.95).float()  # 可信度阈值
                pseudo_labels = pseudo_labels * mask  # 将伪标签中不可信的部分置为 0
                pseudo_labels = pseudo_labels.long()  # 将标签转换为 long 类型

                # 计算损失
                loss = fixmatch_criterion(strong_logits, pseudo_labels, mask)

                # 反向传播与优化
                if not loss.requires_grad:
                    print("Loss does not require gradients!")
                else:
                    loss.backward()

                optimizer.zero_grad()  # 清空梯度
                optimizer.step()  # 更新模型参数

        # 保存模型
        torch.save(model.state_dict(), 'resnet18_model.pth')

    # 测试过程
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = correct / total * 100
    print(f"Test Accuracy（通过测试数据集测试模型准确性）: {test_acc:.2f}%")


def augment_data(unlabeled_images, weak=False):
    # 定义数据增强操作（不再使用 ToTensor，如果图像已是 Tensor）
    transform = transforms.Compose([
        # 在此处根据需要添加其他增强操作，例如随机裁剪、翻转等
    ])

    weak_images = []
    for image in unlabeled_images:
        # 检查 image 是否已经是 Tensor，如果不是才使用 ToTensor()
        if isinstance(image, torch.Tensor):
            weak_images.append(image)  # 如果是 Tensor，直接使用
        else:
            image = transforms.ToTensor()(image)  # 如果是 PIL 图像或 ndarray，转换为 Tensor
            weak_images.append(image)

    # 将图像堆叠成一个批次
    weak_images = torch.stack(weak_images)  # 堆叠成一个批次，形状是 (batch_size, channels, height, width)

    return weak_images


if __name__ == "__main__":
    main()

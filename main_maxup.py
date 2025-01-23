
from utils import SARDataset
import torch.nn.functional as F
from models.network import load_model
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from methods.mixup import mixup_data
import numpy as np
from tqdm import tqdm

# 数据增强示例：包括调整大小、转换为Tensor和归一化，将图像转化为神经网络模型可以处理的格式。
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # 将灰度图转换为3通道RGB图像
    transforms.Resize((224, 224)),  # 调整图像大小为224x224
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 预训练模型的标准化
])

class_names = ['2S1', 'BMP2', 'BRDM_2', 'BTR60', 'BTR70', 'D7', 'T62', 'T72', 'ZIL131', 'ZSU_23_4']

def main():
    # 路径配置
    #
    train_img_dir = './data/MSTAR/mstar-train'  # 有标签的训练数据
    test_img_dir = './data/MSTAR/mstar-test'  # 测试数据
    unlabeled_img_dir = './data/MSTAR/mstar-unlabeled'  # 无标签数据路径

    # 创建训练数据集和测试数据集
    train_dataset = SARDataset(img_dir=train_img_dir, class_names=class_names, transform=transform
                               ,max_size=100)
    # 查看数据集中的图片数量
    print(f"Dataset size（train_dataset）: {len(train_dataset)}")
    test_dataset = SARDataset(img_dir=test_img_dir, class_names=class_names, transform=transform
                              ,max_size=100)

    # 创建无标签数据集
    unlabeled_dataset = SARDataset(img_dir=unlabeled_img_dir, class_names=class_names, transform=transform, is_unlabeled=True
                                   ,max_size=1000)

   #  # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    train_labels = [label for _, label in train_dataloader]  # 只提取标签
    test_labels = [label for _, label in test_dataloader]  # 只提取标签

    # 打印标签的前几个元素进行检查
    print("训练数据标签:", train_labels[:3])
    print("测试数据标签:", test_labels[:3])

    # 加载ResNet18模型
    model =load_model(class_names,False)
    #
    # #检测是否有可用的GPU，如果有则使用GPU，否则使用CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #将模型移动到正确的计算设备上，以便后续的计算在GPU或CPU上进行。
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 有标签数据的损失函数
    #使用 Adam 优化器，学习率为 0.001
    # optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器
    # 创建两个不同的优化器，分别对应有标签和无标签数据的不同学习率
    optimizer_labeled = optim.Adam(model.parameters(), lr=0.1)  # 有标签数据的优化器
    optimizer_unlabeled = optim.Adam(model.parameters(), lr=0.005)  # 无标签数据的优化器

    # 定义学习率调度器
    #使用 StepLR 学习率调度器，每 5 个 epoch 将学习率降低 10%。
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    # 创建两个不同的优化器，分别对应有标签和无标签数据的不同学习率
    scheduler_labeled = optim.lr_scheduler.StepLR(optimizer_labeled, step_size=20, gamma=0.1)
    scheduler_unlabeled = optim.lr_scheduler.StepLR(optimizer_unlabeled, step_size=5, gamma=0.1)

    # 训练过程
    num_epochs = 10  # 设置训练的轮数为 10
    # 设置test验证频率
    validation_frequency = 2  # 每5个epoch进行一次验证
    # 早停策略参数
    patience = 3  # 如果验证损失在5个epoch内没有改善，提前停止训练
    best_val_loss = np.inf  # 初始时设置为正无穷
    epochs_without_improvement = 0  # 跟踪验证损失未改善的轮数

    for epoch in range(num_epochs):
        print(f"训练轮次(start): {epoch + 1} ")

        # 设置模型为训练模式
        model.train()

        # 初始化 running_loss、correct、total：用于记录当前 epoch 的损失、正确预测的数量和总样本数
        running_loss = 0.0
        correct = 0
        total = 0

        # mixup 训练有标签数据
        print("有标签数据训练")
        for (images, labels) in train_dataloader:
            # 从训练数据加载器中加载一批图像和标签
            images, labels = images.to(device), labels.to(device)

            # 清零优化器的梯度
            optimizer_labeled.zero_grad()

            # MixUp数据增强：使用 MixUp 方法生成混合图像和标签
            mixed_images, mixed_labels, lam = mixup_data(images, labels, alpha=1.0)

            # 前向传播
            outputs = model(mixed_images)
            loss = criterion(outputs, mixed_labels)

            # 反向传播
            loss.backward()
            optimizer_labeled.step()

            # 累加损失
            running_loss += loss.item()

            # 获取预测的类别
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)  # 累加总样本数
            correct += (predicted == labels).sum().item()  # 累加正确预测的数量

        # 每个 epoch 输出损失和准确度
        epoch_loss = running_loss / len(train_dataloader)
        epoch_acc = correct / total * 100
        print(f"Epoch（训练轮次） [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy_labeled: {epoch_acc:.2f}%")

        # 使用 MixUp 对无标签数据进行训练
        print("无标签数据训练")
        for unlabeled_images in unlabeled_dataloader:  # 获取无标签图像
            optimizer_unlabeled.zero_grad()
            unlabeled_images = unlabeled_images.to(device)

            # 数据增强：生成弱增强和强增强版本
            weak_images = augment_data(unlabeled_images, weak=True)
            strong_images = augment_data(unlabeled_images, weak=False)

            # 获取弱增强图像的预测
            weak_logits = model(weak_images)

            # 获取强增强图像的预测
            strong_logits = model(strong_images)
            #loss = criterion(strong_logits, pseudo_labels)

            # 计算伪标签：基于弱增强图像的预测概率
            weak_probs = F.softmax(weak_logits, dim=1)
            max_probs, pseudo_labels = weak_probs.max(dim=1)

            # 伪标签过滤：根据最大概率值筛选可信度高的伪标签
            mask = max_probs.ge(0.9).float()  # 可信度阈值
            pseudo_labels = pseudo_labels * mask  # 过滤不可信的伪标签
            pseudo_labels = pseudo_labels.long()

            # MixUp增强：对伪标签数据进行MixUp
            mixed_images, mixed_labels, lam = mixup_data(strong_images, pseudo_labels, alpha=1.0)

            # 计算损失
            strong_logits = model(mixed_images)
            loss = criterion(strong_logits, mixed_labels)

            # 反向传播
            loss.backward()
            optimizer_unlabeled.step()

            # 累加损失
            running_loss += loss.item()

            # 获取预测的类别
            _, predicted = torch.max(strong_logits.data, 1)

            total += mixed_labels.size(0)  # 累加总样本数
            correct += (predicted == mixed_labels).sum().item()  # 累加正确预测的数量

        # 每个 epoch 输出损失和准确度
        epoch_loss = running_loss / len(unlabeled_dataloader)
        epoch_acc = correct / total * 100
        print(f"Epoch（训练轮次） [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy_unlabeled: {epoch_acc:.2f}%")

        # 在每个 epoch 后更新学习率
        # 更新有标签数据的学习率
        scheduler_labeled.step()

        # 更新无标签数据的学习率
        scheduler_unlabeled.step()

        # 保存模型
        torch.save(model.state_dict(), f'model_epoch_{epoch+1}_maxup.pth')

        # 每隔validation_frequency个epoch进行一次验证
        if (epoch + 1) % validation_frequency == 0:
            val_loss, val_acc = test_validate(model, test_dataloader, criterion, device)
            print(f"TestValidation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

            # 早停策略：如果验证损失没有改善，则停止训练
            #通过早停保存的模型是训练过程中性能最稳定、泛化能力最强的模型，因此它通常被认为是最佳的模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                # 保存模型(最佳）
                torch.save(model.state_dict(), f'model_epoch_{epoch + 1}_maxup_best.pth')
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print("Early stopping triggered. No improvement in validation loss.")
                    break

    print("Training finished.")

def test_validate(model, dataloader, criterion, device):
    """
    验证过程函数，执行一个epoch的验证
    参数：
    - model: 神经网络模型
    - dataloader: 验证数据加载器
    - criterion: 损失函数
    - device: 计算设备（CPU或GPU）
    """
    model.eval()  # 将模型设为评估模式
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # 禁止梯度计算，节省内存
        for images, labels in tqdm(dataloader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # 获取模型输出
            loss = criterion(outputs, labels)  # 计算损失

            running_loss += loss.item()  # 累加损失
            _, predicted = torch.max(outputs.data, 1)  # 获取预测的标签
            total += labels.size(0)  # 累计样本数量
            correct += (predicted == labels).sum().item()  # 累计正确预测的样本数

    val_loss = running_loss / len(dataloader)  # 平均验证损失
    val_acc = correct / total * 100  # 验证准确率
    return val_loss, val_acc

def augment_data(unlabeled_images, weak=False):
    # 定义数据增强操作（不再使用 ToTensor，如果图像已是 Tensor）
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 随机裁剪
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 随机调整亮度、对比度、饱和度
    ])

    weak_images = []
    for image in unlabeled_images:
        # 检查 image 是否已经是 Tensor，如果不是才使用 ToTensor()
        if isinstance(image, torch.Tensor):
            weak_images.append(image)  # 如果是 Tensor，直接使用
        else:
            image = transforms.ToTensor()(image)  # 如果是 PIL 图像或 ndarray，转换为 Tensor
            weak_images.append(image)

    # 如果需要弱增强（weak augmentation）
    if weak:
        weak_images = [transform(image) for image in weak_images]
    # 将图像堆叠成一个批次
    weak_images = torch.stack(weak_images)  # 堆叠成一个批次，形状是 (batch_size, channels, height, width)

    return weak_images


if __name__ == "__main__":
    main()

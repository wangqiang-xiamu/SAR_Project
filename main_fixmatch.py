from utils import SARDataset
import torch.nn.functional as F
from models.network import load_model
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from methods.mixup import mixup_data
from methods.fixmatch import fixmatch_criterion

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
    train_dataset = SARDataset(img_dir=train_img_dir, class_names=class_names, transform=transform)
    test_dataset = SARDataset(img_dir=test_img_dir, class_names=class_names, transform=transform)

    # 创建无标签数据集
    unlabeled_dataset = SARDataset(img_dir=unlabeled_img_dir, class_names=class_names, transform=transform, is_unlabeled=True)

   #  # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

   #  # 假设 train_loader 和 test_loader 是您的训练和测试数据加载器
    train_labels = [label for _, label in train_dataloader]  # 只提取标签
    test_labels = [label for _, label in test_dataloader]  # 只提取标签

    # 打印标签的前几个元素进行检查
    print("训练数据标签:", train_labels[:10])
    print("测试数据标签:", test_labels[:10])

    # 加载ResNet18模型
    model =load_model(class_names)

    #检测是否有可用的GPU，如果有则使用GPU，否则使用CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #将模型移动到正确的计算设备上，以便后续的计算在GPU或CPU上进行。
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 有标签数据的损失函数
    #使用 Adam 优化器，学习率为 0.001
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

    # 定义学习率调度器
    #使用 StepLR 学习率调度器，每 5 个 epoch 将学习率降低 10%。
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # 训练过程
    num_epochs=10
    # threshold=0.95 是用于生成伪标签的一个阈值。具体来说，它的作用是在 FixMatch 方法中，
    # 通过设置一个最小的可信度门槛来决定是否将无标签数据的预测作为伪标签。
    # 只有当模型对无标签数据的预测概率高于这个阈值时，才会将其作为有效的伪标签用于训练。
    threshold = 0.95
    for epoch in range(num_epochs):
        print(f"训练轮次(start): {epoch+1} ")

        # 设置模型为训练模式
        model.train()

        # 初始化 running_loss、correct、total：用于记录当前 epoch 的损失、正确预测的数量和总样本数
        running_loss = 0.0
        correct = 0
        total = 0

        # 训练有标签数据（使用 FixMatch 的方式）
        print("有标签数据训练（FixMatch）")
        for (images, labels) in train_dataloader:
            images, labels = images.to(device), labels.to(device)

            # 清零优化器的梯度
            optimizer.zero_grad()

            # 对有标签数据进行弱增强和强增强
            weak_images = augment_data(images, weak=True)  # 假定 augment_data 是您的数据增强方法
            strong_images = augment_data(images, weak=False)

            # 获取弱增强图像的预测
            weak_logits = model(weak_images)
            weak_probs = F.softmax(weak_logits, dim=1)

            # 伪标签生成：使用弱增强的预测作为伪标签
            max_probs, pseudo_labels = weak_probs.max(dim=1)

            # 根据概率过滤伪标签：如果预测概率大于阈值，才将其作为伪标签
            mask = max_probs.ge(threshold).float()
            pseudo_labels = pseudo_labels * mask  # 只保留高可信度的伪标签
            pseudo_labels = pseudo_labels.long()

            # 使用强增强图像和伪标签计算损失
            strong_logits = model(strong_images)
            loss = criterion(strong_logits, pseudo_labels)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 累加损失
            running_loss += loss.item()

            # 获取预测的类别
            _, predicted = torch.max(strong_logits.data, 1)

            total += labels.size(0)  # 累加总样本数
            correct += (predicted == labels).sum().item()  # 累加正确预测的数量

        # 输出有标签数据的损失和准确度
        epoch_loss = running_loss / len(train_dataloader)
        epoch_acc = correct / total * 100
        print(f"Epoch（有标签训练轮次） [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

        # 训练无标签数据（FixMatch）
        print("无标签数据训练（FixMatch）")
        for unlabeled_images in unlabeled_dataloader:
            unlabeled_images = unlabeled_images.to(device)

            # 生成弱增强和强增强版本
            weak_images = augment_data(unlabeled_images, weak=True)  # 假定 augment_data 是您的数据增强方法
            strong_images = augment_data(unlabeled_images, weak=False)

            # 获取弱增强图像的预测
            weak_logits = model(weak_images)
            weak_probs = F.softmax(weak_logits, dim=1)

            # 伪标签生成：使用弱增强的预测作为伪标签
            max_probs, pseudo_labels = weak_probs.max(dim=1)

            # 根据概率过滤伪标签：如果预测概率大于阈值，才将其作为伪标签
            mask = max_probs.ge(threshold).float()
            pseudo_labels = pseudo_labels * mask  # 只保留高可信度的伪标签
            pseudo_labels = pseudo_labels.long()

            # 使用强增强图像和伪标签计算损失
            strong_logits = model(strong_images)
            loss = criterion(strong_logits, pseudo_labels)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 累加损失
            running_loss += loss.item()

            # 获取预测的类别
            _, predicted = torch.max(strong_logits.data, 1)

            total += pseudo_labels.size(0)  # 累加总样本数
            correct += (predicted == pseudo_labels).sum().item()  # 累加正确预测的数量

        # 输出无标签数据训练的损失和准确度
        epoch_loss = running_loss / len(unlabeled_dataloader)
        epoch_acc = correct / total * 100
        print(f"Epoch（无标签训练轮次） [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

        # 在每个 epoch 后更新学习率
        scheduler.step()  # 假定您有学习率调度器（如果没有可以忽略）


        # 保存模型
        torch.save(model.state_dict(), f'model_epoch_{epoch+1}_fixmatch.pth')

    # 测试过程
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_dataloader:
            print(f"labels: {labels}")  # 打印标签，查看标签是否正确加载
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            print(f"predicted: {predicted}")  # 打印模型的预测值
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = correct / total * 100
    print(f"Test Accuracy（通过测试数据集测试模型准确性）: {test_acc:.2f}%")

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


from utils import SARDataset
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from methods.mixup import mixup_data
from methods.fixmatch import fixmatch_criterion
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
#测试
#class_names = ['2S1', 'BMP2']

def main():
    # 路径配置
    #
    train_img_dir = './data/MSTAR/mstar-train'  # 有标签的训练数据
    test_img_dir = './data/MSTAR/mstar-test'  # 测试数据
    unlabeled_img_dir = './data/MSTAR/mstar-unlabeled'  # 无标签数据路径
    #
    # train_img_dir = './data/MSTAR/mstar-train-test'  # 有标签的训练数据
    # test_img_dir = './data/MSTAR/mstar-test-test'  # 测试数据
    # unlabeled_img_dir = './data/MSTAR/mstar-unlabeled'  # 无标签数据路径

    # 创建训练数据集和测试数据集
    train_dataset = SARDataset(img_dir=train_img_dir, class_names=class_names, transform=transform)
    test_dataset = SARDataset(img_dir=test_img_dir, class_names=class_names, transform=transform)

    # 创建无标签数据集
    unlabeled_dataset = SARDataset(img_dir=unlabeled_img_dir, class_names=class_names, transform=transform, is_unlabeled=True)

   #  # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
   #测试
   #  train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)
   #  test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)
   #  unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)
   #
   #  # 假设 train_loader 和 test_loader 是您的训练和测试数据加载器
    train_labels = [label for _, label in train_dataloader]  # 只提取标签
    test_labels = [label for _, label in test_dataloader]  # 只提取标签

    # 打印标签的前几个元素进行检查
    print("训练数据标签:", train_labels[:10])
    print("测试数据标签:", test_labels[:10])

    # 加载ResNet18模型
    #model =load_model(class_names,false)
    #测试轻量级mobilenet_v2
    # 加载MobileNetV2预训练模型
    model = models.mobilenet_v2(weights=None)
    # 获取原始模型最后一层的输入特征数
    num_ftrs = model.classifier[1].in_features  # 原来的输出特征数
    # 修改最后一层，将输出类别数设置为 len(class_names)
    model.classifier[1] = nn.Linear(num_ftrs, len(class_names))  # 更新为我们任务的类别数

    #检测是否有可用的GPU，如果有则使用GPU，否则使用CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #将模型移动到正确的计算设备上，以便后续的计算在GPU或CPU上进行。
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 有标签数据的损失函数
    #使用 Adam 优化器，学习率为 0.001
    optimizer_labeled = optim.Adam(model.parameters(), lr=0.01)  # 有标签数据的优化器
    optimizer_unlabeled = optim.Adam(model.parameters(), lr=0.005)  # 无标签数据的优化器

    # 定义学习率调度器
    #使用 StepLR 学习率调度器，每 5 个 epoch 将学习率降低 10%。
    scheduler_labeled = optim.lr_scheduler.StepLR(optimizer_labeled, step_size=20, gamma=0.1)
    scheduler_unlabeled = optim.lr_scheduler.StepLR(optimizer_unlabeled, step_size=5, gamma=0.1)

    # 训练过程
    #有标签数据
    #MixUp数据增强
    #设置训练的轮数为 10。
    # 训练过程
    num_epochs = 3  # 设置训练的轮数为 10
    # 设置test验证频率
    validation_frequency = 5  # 每5个epoch进行一次验证
    # 早停策略参数
    patience = 5  # 如果验证损失在5个epoch内没有改善，提前停止训练
    best_val_loss = np.inf  # 初始时设置为正无穷
    epochs_without_improvement = 0  # 跟踪验证损失未改善的轮数

    for epoch in range(num_epochs):
        print(f"训练轮次(start): {epoch+1} ")
        #设置模型为训练模式。
        model.train()
        #初始化running_loss、correct、total：用于记录当前 epoch 的损失、正确预测的数量和总样本数。
        running_loss = 0.0
        correct = 0
        total = 0
        print("有标签数据训练")
        for (images, labels) in train_dataloader:
            # 从训练数据加载器中加载一批图像和标签
            images, labels = images.to(device), labels.to(device)
            # 清零优化器的梯度
            optimizer_labeled.zero_grad()

            # MixUp数据增强
            #使用 MixUp 数据增强方法，生成混合图像和标签。
            mixed_images, mixed_labels, lam = mixup_data(images, labels, alpha=1.0)
            # 前向传播
            #通过模型进行前向传播，计算损失。
            outputs = model(mixed_images)
            loss = criterion(outputs, mixed_labels)
            #print("Loss:", loss.item())  # 输出损失

            # 反向传播
            #反向传播计算梯度，并更新模型参数。`
            loss.backward()
            optimizer_labeled.step()
            #running_loss 是一个累加器，用于记录每个 epoch 中所有批次的总损失。
            #loss.item() 获取当前批次的损失值，并将其累加到 running_loss 中。
            # loss 是模型输出与真实标签之间的损失，item() 将 PyTorch 张量转换为 Python 数字。
            running_loss += loss.item()
            #outputs 是模型对输入数据的预测结果，通常是一个包含各类别概率分布的张量（例如，大小为 [batch_size, num_classes]）。
            # torch.max(outputs.data, 1) 计算每个样本在输出中最大概率对应的类别索引。这里 1 表示沿着类别轴（即第二维度）选择最大值。
            # 结果 predicted 是模型预测的类别索引。
            # outputs.data 取出模型的原始预测值，不需要计算梯度。
            _, predicted = torch.max(outputs.data, 1)
            labels.size(0)
            # 返回当前批次的样本数量（即批次的大小）。
            # total 累加当前epoch 中所有批次的样本数，用于计算准确率的分母
            total += labels.size(0)
            # (predicted == labels) 是一个布尔张量，表示模型预测是否正确。
            # sum() 统计预测正确的样本数。item() 将结果转换为 Python 数字，
            # 方便累加到 correct 中。correct 累加当前 epoch 中所有批次预测正确的样本数。
            correct += (predicted == labels).sum().item()

        #每个 epoch（训练轮次） 输出损失和准确度
        #损失率会随着训练轮次（epoch）的增加而逐渐减小
        epoch_loss = running_loss / len(train_dataloader)
        epoch_acc = correct / total * 100
        #print(f"Epoch（训练轮次）[{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")


        # 使用FixMatch进行无标签数据训练
        print("无标签数据训练")
        # 是一个上下文管理器，
        # 表示在其范围内不计算梯度。它通常用于推理（推断），以减少内存开销和计算时间，
        # 因为模型的输出不需要用于反向传播和梯度计算。
        # 在此处，我们只是通过模型进行预测，因此不需要计算梯度。
        for unlabeled_images in unlabeled_dataloader:  # 直接获取图像
            # 清零优化器的梯度
            optimizer_unlabeled.zero_grad()

            # 将批次中的无标签图像unlabeled_images移动到计算设备上（GPU 或CPU）
            unlabeled_images = unlabeled_images.to(device)

                # augment_data 是一个数据增强函数，
                # 它会根据参数 weak=True 为无标签数据生成“弱增强”版本的图像。
                # 弱增强通常是指不太强的图像变换，比如轻微的裁剪或颜色变换，目的是保持图像的原始结构。
            weak_images = augment_data(unlabeled_images, weak=True)
                # augment_data 当 weak=False 表示生成“强增强”版本的图像。
                # 强增强通常包含更多变化，比如更强的裁剪、旋转或颜色变换，
                # 目的是让模型看到更加多样化的图像，以提高其鲁棒性。
            strong_images = augment_data(unlabeled_images, weak=False)

                # 获取模型的预测
                # 将经过弱增强的图像 weak_images 输入到模型 model 中，
                # 得到预测的 logits（预测得分）。weak_logits 是一个包含每个类别得分的张量，
                # 形状通常为 [batch_size, num_classes]。
            weak_logits = model(weak_images)  # 计算弱增强的预测

                # 将经过强增强的图像 strong_images 输入到模型 model 中，得到预测的 logits（预测得分）。
                # strong_logits 是与 weak_logits 形状相同的张量，但它是基于强增强图像的预测。
            strong_logits = model(strong_images)  # 计算强增强的预测

                # requires_grad_() 是 PyTorch 中设置 Tensor 是否需要计算梯度的函数。
                # 此处，它被用来确保 weak_logits 的计算图包含在反向传播中，
                # 即使其本身并不是在训练过程中优化的参数。
                # 通常，weak_logits 用于计算伪标签，所以需要计算梯度。
            weak_logits.requires_grad_()  # 确保 weak_logits 需要梯度

                #同样，strong_logits 也需要计算梯度，
                # 因为我们将在后续的步骤中使用 strong_logits 来计算损失。
            strong_logits.requires_grad_()  # 确保 strong_logits 需要梯度

                # 计算伪标签
                #F.softmax：这是 PyTorch 的一个函数，
                # 用来对 weak_logits 进行 Softmax 操作，
                # 得到每个类别的概率分布。dim=1 表示在类别维度（即每一行）进行 Softmax 操作。
                # weak_probs 是每个样本在各个类别上的概率分布。
            weak_probs = F.softmax(weak_logits, dim=1)

                # max_probs 表示每个样本的最大概率值。
                # pseudo_labels 表示每个样本的伪标签，即最大概率对应的类别。
            max_probs, pseudo_labels = weak_probs.max(dim=1)

                # 伪标签过滤
                # max_probs.ge(0.95)：ge 是“greater than or equal”的缩写
                # 表示计算每个样本的最大概率是否大于等于 0.95。这样可以得到一个布尔张量，
                # 表示哪些伪标签是高可信度的。
                #.float()：将布尔值转换为浮动值，True 会变成 1.0，False 会变成 0.0。
                # 得到的 mask 用来过滤伪标签中不可信的部分。
            mask = max_probs.ge(0.9).float()  # 可信度阈值
            pseudo_labels = pseudo_labels * mask  # 将伪标签中不可信的部分置为 0
            pseudo_labels = pseudo_labels.long()  # 将标签转换为 long 类型

                # 计算损失
                #fixmatch_criterion：这是一个自定义的损失函数，用于计算基于伪标签的损失。
                # 它使用强增强图像的预测 strong_logits 和生成的伪标签 pseudo_labels 来计算损失，
                # 并且通过 mask 来应用过滤器，仅计算高可信度的伪标签部分。
            loss = fixmatch_criterion(strong_logits, pseudo_labels, mask)

                #backward()：进行反向传播计算梯度。根据损失 loss 对模型的所有参数计算梯度。
            loss.backward()
                #zero_grad()：在每次反向传播前，我们需要清空之前计算的梯度，否则梯度会累积。此行代码在每次优化步骤前清空梯度。
            optimizer_unlabeled.zero_grad()  # 清空梯度
                #step()：通过优化器更新模型的参数。它会根据计算出来的梯度来调整模型参数，以最小化损失函数。
            optimizer_unlabeled.step()  # 更新模型参数
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)

            labels.size(0)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

            # 每个 epoch（训练轮次） 输出损失和准确度
            # 损失率会随着训练轮次（epoch）的增加而逐渐减小
        epoch_loss = running_loss / len(train_dataloader)
        epoch_acc = correct / total * 100
        print(f"Epoch（训练轮次） [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

        # 在每个epoch后更新学习率

        # 更新有标签数据的学习率
        scheduler_labeled.step()

        # 更新无标签数据的学习率
        scheduler_unlabeled.step()

        # 保存模型
        torch.save(model.state_dict(), f'model_epoch_mobilenet_v2_{epoch+1}.pth2')

        # 每隔validation_frequency个epoch进行一次验证
        if (epoch + 1) % validation_frequency == 0:
            val_loss, val_acc = test_validate(model, test_dataloader, criterion, device)
            print(f"TestValidation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

            # 早停策略：如果验证损失没有改善，则停止训练
            # 通过早停保存的模型是训练过程中性能最稳定、泛化能力最强的模型，因此它通常被认为是最佳的模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                # 保存模型(最佳）
                torch.save(model.state_dict(), f'model_epoch_mobilenet_v2_{epoch + 1}_best.pth')
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

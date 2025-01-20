# methods/mixup.py
import torch
import numpy as np

def mixup_data(x, y, alpha=1.0):
    """
    实现 MixUp 数据增强
    :param x: 输入数据（图像）
    :param y: 真实标签
    :param alpha: Beta分布的参数，控制MixUp强度
    :return: 混合后的数据和标签
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)  # 从Beta分布中采样lambda
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)  # 打乱数据顺序

    mixed_x = lam * x + (1 - lam) * x[index, :]  # 线性组合图像
    y_a, y_b = y, y[index]  # 混合标签
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    MixUp损失函数
    :param criterion: 损失函数
    :param pred: 模型预测结果
    :param y_a, y_b: 两种标签
    :param lam: MixUp的lambda
    :return: 损失值
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

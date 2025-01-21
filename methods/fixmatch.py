import torch
import torch.nn.functional as F


def sharpen(predictions, T=0.5):
    """
    伪标签锐化函数，调整伪标签的确信度
    :param predictions: 预测的概率分布
    :param T: 温度参数，值越小伪标签越确信
    :return: 锐化后的概率分布
    """
    predictions = predictions ** (1 / T)  # 温度调整
    return predictions / predictions.sum(dim=1, keepdim=True)  # 归一化


def fixmatch_criterion(strong_logits, pseudo_labels, mask, lambda_u=1.0):
    """
    计算 FixMatch 损失函数，考虑了强增强数据和伪标签的可信度。
    """
    # 计算交叉熵损失
    cross_entropy_loss = F.cross_entropy(strong_logits, pseudo_labels)

    # 只对可信伪标签部分计算损失
    loss = cross_entropy_loss * mask
    return loss.mean() * lambda_u  # 对损失加权并返回

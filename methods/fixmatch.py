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


def fixmatch_criterion(weak_logits, strong_logits, labels=None, threshold=0.95, T=0.5):
    """
    FixMatch损失函数
    :param weak_logits: 弱增强预测结果
    :param strong_logits: 强增强预测结果
    :param labels: 真实标签（监督部分），可选
    :param threshold: 伪标签确信度阈值，只有高于该阈值的伪标签才会用于计算损失
    :param T: 温度参数，用于锐化伪标签
    :return: 损失值
    """
    # 计算弱增强的概率分布
    probs = torch.softmax(weak_logits, dim=1)

    # 获取伪标签（最大概率的标签）和它的概率
    max_probs, pseudo_labels = torch.max(probs, dim=1)

    # 使用温度缩放伪标签的确信度
    probs = sharpen(probs, T)  # 锐化处理

    # 筛选出确信度高于阈值的伪标签
    mask = max_probs.ge(threshold).float()

    # 计算强增强预测的损失
    loss = F.cross_entropy(strong_logits, pseudo_labels, reduction='none')
    loss = (loss * mask).mean()  # 使用高确信度伪标签计算损失

    return loss

import torch
import torch.nn.functional as F
# sharpen：通过调节温度参数T来增强伪标签的确信度。
# threshold：用于确定哪些伪标签是“可信”的，只有预测确信度高于此阈值的伪标签才会被使用。
def sharpen(predictions, T=0.5):
    """
    伪标签锐化函数，调整伪标签的确信度
    :param predictions: 预测的概率分布
    :param T: 温度参数，值越小伪标签越确信
    :return: 锐化后的概率分布
    """
    predictions = predictions ** (1 / T)
    return predictions / predictions.sum(dim=1, keepdim=True)

def fixmatch_criterion(weak_logits, strong_logits, labels, threshold=0.95):
    """
    FixMatch损失函数
    :param weak_logits: 弱增强预测结果
    :param strong_logits: 强增强预测结果
    :param labels: 真实标签（监督部分）
    :param threshold: 伪标签确信度阈值，只有高于该阈值的伪标签才会用于计算损失
    :return: 损失值
    """
    probs = torch.softmax(weak_logits, dim=1)  # 计算弱增强预测概率
    max_probs, pseudo_labels = torch.max(probs, dim=1)  # 获取最大概率的伪标签
    mask = max_probs.ge(threshold).float()  # 根据阈值筛选伪标签

    loss = F.cross_entropy(strong_logits, pseudo_labels, reduction='none')
    loss = (loss * mask).mean()  # 使用高确信度伪标签计算损失
    return loss

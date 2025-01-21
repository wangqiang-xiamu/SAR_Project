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


def fixmatch_criterion(weak_logits, strong_logits, threshold=0.95, T=0.5):
    """
    FixMatch损失函数
    :param weak_logits: 弱模型的输出
    :param strong_logits: 强模型的输出
    :param threshold: 阈值，用于筛选高置信度的伪标签
    :param T: 温度缩放系数
    :return: 计算出的损失
    """
    # 对弱模型的输出应用 softmax 和温度缩放
    weak_probs = torch.softmax(weak_logits / T, dim=1)  # 应用 temperature scaling 和 softmax
    print(f"weak_probs shape: {weak_probs.shape}")
    print(f"weak_probs sample: {weak_probs[0]}")  # 打印一个样本的概率分布

    # 获取最大概率的索引（伪标签）
    max_probs, pseudo_labels = torch.max(weak_probs, dim=1)
    print(f"max_probs shape: {max_probs.shape}")
    print(f"pseudo_labels shape before mask: {pseudo_labels.shape}")

    # 创建掩码：如果最大概率高于阈值，则保留该标签
    mask = max_probs.ge(threshold).float()
    pseudo_labels = pseudo_labels * mask.long()  # 用掩码更新伪标签

    print(f"pseudo_labels shape after mask: {pseudo_labels.shape}")
    print(f"pseudo_labels content after mask: {pseudo_labels}")

    # 确保 pseudo_labels 是 Long 类型并且是一维的，符合 cross_entropy 的要求
    pseudo_labels = pseudo_labels.long()  # 确保是 Long 类型
    if pseudo_labels.dim() != 1:
        pseudo_labels = pseudo_labels.view(-1)  # 强制将伪标签转为一维

    # 确保 strong_logits 的形状是 [batch_size, num_classes]
    assert strong_logits.shape[0] == pseudo_labels.shape[0], "Batch size mismatch"

    # 输出检查
    print(f"strong_logits shape: {strong_logits.shape}")
    print(f"pseudo_labels shape: {pseudo_labels.shape}")
    print(f"pseudo_labels content: {pseudo_labels}")

    # 计算强模型的损失
    # cross_entropy 期望目标是长整型的类别索引，因此我们不需要将其转为 one-hot 编码
    # 假设你有 10 个类别
    num_classes = 10

    # 强制调整 strong_logits 的形状为 [batch_size, num_classes]
    # 你需要确保 strong_logits 是每个样本的类 logits，而不是单一的标量值
    if strong_logits.dim() == 1:  # 如果它是 [batch_size]，我们需要扩展它
        strong_logits = strong_logits.unsqueeze(1).expand(-1, num_classes)  # 变为 [batch_size, num_classes]

    # 判断 pseudo_labels 的类型
    if pseudo_labels.dim() == 1 or pseudo_labels.max() < 1:
        # 假设是类别索引（long 类型）
        # 目标标签是整数类型 (Long)

        strong_logits = strong_logits.float()  # 确保 strong_logits 是浮动点类型

        pseudo_labels = pseudo_labels.long()  # 确保 pseudo_labels 是 Long 类型
        loss = F.cross_entropy(strong_logits, pseudo_labels.long(), reduction='none')
    else:
        # 假设是概率分布（float 类型）
        # 目标标签是浮动类型 (Float)
        strong_logits = strong_logits.float()  # 确保 strong_logits 是浮动点类型

        pseudo_labels = pseudo_labels.long()  # 确保 pseudo_labels 是 Long 类型

        loss = F.cross_entropy(strong_logits, pseudo_labels, reduction='none')

    # 按照掩码加权损失
    loss = (loss * mask).mean()  # 只保留可信度较高的部分

    return loss


# 模拟数据（示例）
batch_size = 32
num_classes = 10

# 模拟弱模型和强模型的输出 logits
weak_logits = torch.randn(batch_size, num_classes)  # 假设有 32 个样本，10 类
strong_logits = torch.randn(batch_size, num_classes)

# 调用损失函数
loss = fixmatch_criterion(weak_logits, strong_logits, threshold=0.7, T=0.5)
print(f"Loss: {loss.item()}")

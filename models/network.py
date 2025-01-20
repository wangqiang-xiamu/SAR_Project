import torch
import torch.nn as nn
from torchvision import models


class SARNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SARNet, self).__init__()

        # 使用预训练模型ResNet18，不加载预训练权重
        self.resnet18 = models.resnet18(pretrained=False)

        # 修改ResNet18的最后一层，全连接层输出类别数
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet18(x)  # 直接返回ResNet18的输出

import torch
import torch.nn as nn
import torch.nn.functional as F
# 卷积层（conv1 和 conv2）：用于从图像中提取特征。
# fc1 和 fc2：全连接层，将提取的特征映射到类别空间。
# 激活函数（ReLU）：在卷积和全连接层后使用ReLU进行非线性变换。
class SARNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SARNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 56 * 56, 1024)  # 对应224x224图像
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 第一层卷积
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # 第二层卷积 + 池化
        x = x.view(x.size(0), -1)  # 展平
        x = F.relu(self.fc1(x))  # 第一层全连接
        x = self.fc2(x)  # 输出层
        return x

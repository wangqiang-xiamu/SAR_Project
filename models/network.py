
import torch
import torch.nn as nn
from torchvision import models

# 定义模型加载函数
def load_model(class_names, model_path='resnet18_model.pth'):
    # 加载预训练的ResNet18模型
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # 修改输出层以匹配你的类别数量
    model.fc = nn.Linear(model.fc.in_features, len(class_names))

    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))

    # 设置为评估模式
    model.eval()

    return model

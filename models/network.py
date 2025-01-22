import torch
import os
import torch.nn as nn
from torchvision import models

# 定义模型加载函数
def load_model(class_names, model_path):
    if not model_path:
        print("初始化模型")
        # 加载预训练的ResNet18模型
        model = models.resnet18(weights=None)  # 不加载预训练权重
        return model
    model = models.resnet18(weights=None)  # 不加载预训练权重
    # 获取 ResNet 的最后一层 (fc)
    num_ftrs = model.fc.in_features

    # 修改最后一层，确保输出类别数与当前任务的类别数一致
    model.fc = nn.Linear(num_ftrs, len(class_names))  # 10 类别

    # 加载训练好的权重，忽略最后一层
    state_dict = torch.load(model_path)
    # Remove the last layer parameters (fc layers)
    state_dict.pop('fc.weight', None)
    state_dict.pop('fc.bias', None)

    model.load_state_dict(state_dict, strict=False)  # 只加载匹配的部分

    model.eval()  # 切换到评估模式
    print("加载模型文件成功")
    return model


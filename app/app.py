from flask import Flask, request, jsonify
from PIL import Image
import torch
from torchvision import transforms
from models.network import SARNet  # 导入修改后的SARNet类

# Web接口
app = Flask(__name__)

# 加载训练好的模型
model = SARNet(num_classes=10)  # 根据类别数创建SARNet模型
model.load_state_dict(torch.load('resnet18_model.pth', map_location=torch.device('cpu')))  # 加载训练好的权重
model.eval()  # 设置为评估模式

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为224x224
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 预训练模型标准化
])

# /predict 路由：接收POST请求，处理图像并返回预测结果
@app.route('/predict', methods=['POST'])
def predict():
    # 检查请求中是否包含文件
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # 打开并预处理图像
    image = Image.open(file.stream).convert("RGB")  # 确保图像为RGB格式
    image = transform(image).unsqueeze(0)  # 扩展维度，变为[1, 3, 224, 224]

    # 禁用梯度计算，进行推理
    with torch.no_grad():
        output = model(image)  # 前向传播
        _, predicted = torch.max(output, 1)  # 获取最大概率的类别
        prediction = predicted.item()  # 获取类别编号

    return jsonify({'prediction': prediction})  # 返回预测结果

if __name__ == '__main__':
    app.run(debug=True)

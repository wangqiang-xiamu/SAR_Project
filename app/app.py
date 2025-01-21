from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
from models.network import load_model

# Web接口
app = Flask(__name__)

# 类别名称（根据你的训练数据）
class_names = ['2S1', 'BMP2', 'BRDM_2', 'BTR60', 'BTR70', 'D7', 'T62', 'T72', 'ZIL131', 'ZSU_23_4']

# 加载训练好的模型
model = load_model(class_names, model_path='resnet18_model.pth')

# 图像预处理
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # 将灰度图转换为3通道RGB图像
    transforms.Resize((224, 224)),  # 调整图像大小为224x224
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 预训练模型标准化
])

@app.route('/')
def home():
    # 渲染HTML页面
    return render_template('upload_form.html')

# /predict 路由：接收POST请求，处理图像并返回预测结果
@app.route('/predict', methods=['POST'])
def predict():
    # 检查请求中是否包含文件
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # 打开并预处理图像
        image = Image.open(file.stream).convert("RGB")  # 确保图像为RGB格式
        image = transform(image).unsqueeze(0)  # 扩展维度，变为[1, 3, 224, 224]

        # 禁用梯度计算，进行推理
        with torch.no_grad():
            output = model(image)  # 前向传播
            _, predicted = torch.max(output, 1)  # 获取最大概率的类别
            prediction = predicted.item()  # 获取类别编号

        # 获取预测类别的名称
        predicted_class = class_names[prediction]
        # 返回结果页面，显示识别信息
        return render_template('result.html', class_name=predicted_class)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

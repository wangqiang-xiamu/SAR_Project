import torch
from torchvision import transforms
from flask import Flask, request,render_template
from PIL import Image
import os
from models.network import load_model

# 初始化Flask应用
app = Flask(__name__)

# 设置类名（与训练时一致）
class_names = ['2S1', 'BMP2', 'BRDM_2', 'BTR60', 'BTR70', 'D7', 'T62', 'T72', 'ZIL131', 'ZSU_23_4']

# 预处理函数（与训练时一致）
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # 将灰度图转换为3通道RGB图像
    transforms.Resize((224, 224)),  # 调整图像大小为224x224
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 预训练模型的标准化
])


# 加载模型函数
def load_trained_model(model_path='../model_epoch_3_best.pth'):
    model = load_model(class_names, False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设为评估模式
    return model


# 加载预训练模型
model = load_trained_model('../model_epoch_3_best.pth')


# 预测函数
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')  # 打开图像并转为RGB
    image = transform(image).unsqueeze(0)  # 应用预处理并增加batch维度

    # 选择设备（GPU/CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device)

    # 前向传播
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

    # 获取预测类别
    predicted_class = class_names[predicted.item()]
    return predicted_class


@app.route('/')
def home():
    # 渲染HTML页面
    return render_template('upload_form.html')

# 设置路由（API接口）
@app.route('/predict', methods=['POST'])
def predict():
    # 获取上传的图片文件
    file = request.files['file']

    # 保存临时文件
    image_path = os.path.join('temp', file.filename)
    file.save(image_path)

    # 进行预测
    predicted_class = predict_image(image_path)

    # 返回预测结果
    # 返回结果页面，显示识别信息
    return render_template('result.html', class_name=predicted_class)



# 启动Flask应用
if __name__ == '__main__':
    if not os.path.exists('temp'):
        os.makedirs('temp')
    app.run(debug=True)

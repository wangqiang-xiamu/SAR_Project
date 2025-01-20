from flask import Flask, request, jsonify
from PIL import Image
import torch
from torchvision import transforms
from models.network import SARNet

# Web接口
app = Flask(__name__)

# 加载训练好的模型
model = SARNet(num_classes=10)
model.load_state_dict(torch.load('sar_model_final.pth'))
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# /predict 路由：接收POST请求，处理图像并返回预测结果
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    image = Image.open(file.stream)
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        prediction = predicted.item()

    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(debug=True)

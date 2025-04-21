import torch
from torchvision import transforms
from PIL import Image
from train_driver_teacher_test2 import *

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'save_teacher/model.pth'  
model = torch.load(model_path, map_location=device)  # 加载模型
model.eval()  # 设置为评估模式
model.to(device)

# 定义数据预处理
transform_test = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图片大小
    transforms.CenterCrop(224),    # 中心裁剪
    transforms.ToTensor(),         # 转换为张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化
])

# 加载单张图片
image_path = 'data/test/c0/img_12470.jpg'  # 替换为你的图片路径
image = Image.open(image_path)  # 打开图片
image = transform_test(image).unsqueeze(0).to(device)  # 预处理并添加 batch 维度

# 预测
with torch.no_grad():  # 禁用梯度计算
    outputs = model(image)  # 前向传播
    output_concat = outputs[-1]  # 提取组合输出
    _, predicted = torch.max(output_concat, 1)  # 获取预测类别

# 显示预测结果
class_names = ['class1', 'class2', 'class3', 'class4', 'class5',  # 替换为你的类别名称
               'class6', 'class7', 'class8', 'class9', 'class10']
print(f'Predicted Class: {class_names[predicted.item()]}')

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from train_driver_teacher_test2 import *

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'save_teacher/model.pth'  # 替换为你的模型路径
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

# 加载测试数据集
testset = ImageFolder(root='data/test', transform=transform_test)  # 替换为你的测试集路径
testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=4)

# 测试模型性能
correct = 0
total = 0

with torch.no_grad():  # 禁用梯度计算
    for inputs, targets in testloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)  # 前向传播
        output_concat = outputs[-1]  # 提取组合输出
        _, predicted = torch.max(output_concat, 1)  # 获取预测类别
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

# 计算并打印测试集准确率
accuracy = 100. * correct / total
print(correct,total)

print(f'Test Accuracy: {accuracy:.2f}%')


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

# 文件路径 - 请确保这个路径指向您的结果文件
file_path = 'Image-based Recognition/save_teacher_test/results_train.txt'

# 解析训练数据
data = []
with open(file_path, 'r') as f:
    for line in f:
        parts = line.strip().split('|')
        iteration = int(parts[0].split()[1])
        acc = float(parts[1].split('=')[1].strip())
        loss = float(parts[2].split('=')[1].strip())
        
        # 使用正则表达式提取损失值，以处理可能的格式变化
        loss1 = float(re.search(r'Loss1: ([\d.]+)', parts[3]).group(1))
        loss2 = float(re.search(r'Loss2: ([\d.]+)', parts[4]).group(1))
        loss3 = float(re.search(r'Loss3: ([\d.]+)', parts[5]).group(1))
        loss_concat = float(re.search(r'Loss_concat: ([\d.]+)', parts[6]).group(1))
        
        data.append([iteration, acc, loss, loss1, loss2, loss3, loss_concat])

df = pd.DataFrame(data, columns=['Iteration', 'Accuracy', 'Total Loss', 
                                'Loss1', 'Loss2', 'Loss3', 'Loss Concat'])

# 设置图表样式
plt.style.use('ggplot')

# 创建一个包含两个子图的图表
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# 1. 准确率曲线
ax1.plot(df['Iteration'], df['Accuracy'], 'o-', color='#2C7BB6', linewidth=2, markersize=6)
ax1.set_title('Training Accuracy over Iterations', fontsize=16)
ax1.set_xlabel('Iteration', fontsize=14)
ax1.set_ylabel('Accuracy (%)', fontsize=14)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.set_ylim(0, 105)

# 添加平滑趋势线
z = np.polyfit(df['Iteration'], df['Accuracy'], 3)
p = np.poly1d(z)
ax1.plot(df['Iteration'], p(df['Iteration']), 'r--', linewidth=1.5, 
         label='Trend (Polynomial Fit)')
ax1.legend()

# 2. 损失曲线
ax2.plot(df['Iteration'], df['Total Loss'], 'o-', color='#D95F02', linewidth=2, 
         markersize=6, label='Total Loss')
ax2.plot(df['Iteration'], df['Loss1'], 's-', color='#7570B3', linewidth=1.5, 
         markersize=4, label='Loss1 (First Classifier)')
ax2.plot(df['Iteration'], df['Loss2'], '^-', color='#E7298A', linewidth=1.5, 
         markersize=4, label='Loss2 (Second Classifier)')
ax2.plot(df['Iteration'], df['Loss3'], 'D-', color='#66A61E', linewidth=1.5, 
         markersize=4, label='Loss3 (Third Classifier)')
ax2.plot(df['Iteration'], df['Loss Concat'], 'X-', color='#E6AB02', linewidth=1.5, 
         markersize=4, label='Loss Concat (Combined Classifier)')

ax2.set_title('Training Loss Components over Iterations', fontsize=16)
ax2.set_xlabel('Iteration', fontsize=14)
ax2.set_ylabel('Loss Value', fontsize=14)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend(loc='upper right', fontsize=10)

# 对Y轴使用对数刻度以便更好地查看后期的变化
ax2.set_yscale('log')

# 创建第三个图表，只显示总损失
plt.figure(figsize=(12, 5))
plt.plot(df['Iteration'], df['Total Loss'], 'o-', color='#D95F02', linewidth=2, markersize=6)
plt.title('Total Training Loss over Iterations', fontsize=16)
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Loss Value', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

# 添加平滑趋势线
z = np.polyfit(df['Iteration'], df['Total Loss'], 3)
p = np.poly1d(z)
plt.plot(df['Iteration'], p(df['Iteration']), 'b--', linewidth=1.5, 
         label='Trend (Polynomial Fit)')
plt.legend()

# 保存图表
plt.tight_layout()
fig.savefig('training_curves_combined.png', dpi=300, bbox_inches='tight')
plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')

plt.show()
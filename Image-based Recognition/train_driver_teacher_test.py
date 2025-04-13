from __future__ import print_function
import torch.optim as optim
import torch.backends.cudnn as cudnn
import timm
import os
import random
import imgaug.augmenters as iaa
from utils_Progressive import *
from tqdm import tqdm
import torchvision
import torch.nn as nn
import torch
import numpy as np
import torchvision.transforms as transforms



# 假设 BasicConv 类已经在 utils_Progressive 中定义
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Features(nn.Module):
    def __init__(self, net_layers):
        super(Features, self).__init__()
        self.net_layer_0 = nn.Sequential(net_layers[0])
        self.net_layer_1 = nn.Sequential(net_layers[1])
        self.net_layer_2 = nn.Sequential(net_layers[2])
        self.net_layer_3 = nn.Sequential(net_layers[3])
        self.net_layer_4 = nn.Sequential(*net_layers[4])
        self.net_layer_5 = nn.Sequential(*net_layers[5])
        self.net_layer_6 = nn.Sequential(*net_layers[6])
        self.net_layer_7 = nn.Sequential(*net_layers[7])

    def forward(self, x):
        x = self.net_layer_0(x)
        x = self.net_layer_1(x)
        x = self.net_layer_2(x)
        x = self.net_layer_3(x)
        x = self.net_layer_4(x)
        x1 = self.net_layer_5(x)
        x2 = self.net_layer_6(x1)
        x3 = self.net_layer_7(x2)
        return x1, x2, x3

# 修改 img_progressive 函数以确保返回正确的张量
def img_progressive(x, limit, p=0.5):
    if random.random() < p:
        # 保存原始形状和设备
        original_shape = x.shape
        device = x.device

        # 转换为numpy进行增强
        x_np = x.cpu().numpy()
        x_np = np.transpose(x_np, (0, 2, 3, 1))  # BCHW -> BHWC

        # 将数据从 float32 类型转换为 uint8 类型
        # 因为 imgaug 通常期望输入为 uint8 类型
        x_np_uint8 = (x_np * 255).astype(np.uint8)

        # 应用亮度增强
        aug = iaa.MultiplyBrightness((1 - limit, 1 + limit))
        x_np_augmented = aug(images=x_np_uint8)

        # 将增强后的 uint8 数据转换回 float32 类型
        x_np = x_np_augmented.astype(np.float32) / 255

        # 转回 PyTorch 张量
        x = torch.from_numpy(np.transpose(x_np, (0, 3, 1, 2)))  # BHWC -> BCHW
        x = x.to(device)

        # 确保形状一致
        assert x.shape == original_shape, f"Shape mismatch: {x.shape} vs {original_shape}"

    return x

class Network_Wrapper(nn.Module):
    def __init__(self, net_layers):
        super().__init__()
        self.Features = Features(net_layers)

        self.max_pool1 = nn.MaxPool2d(kernel_size=28, stride=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=14, stride=1)
        self.max_pool3 = nn.MaxPool2d(kernel_size=7, stride=1)

        self.conv_block1 = nn.Sequential(
            BasicConv(512, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, 10)
        )

        self.conv_block2 = nn.Sequential(
            BasicConv(1024, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, 10),
        )

        self.conv_block3 = nn.Sequential(
            BasicConv(2048, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, 10),
        )

        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(1024 * 3),
            nn.Linear(1024 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x1, x2, x3 = self.Features(x)

        x1_ = self.conv_block1(x1)
        x1_ = self.max_pool1(x1_)
        x1_f = x1_.view(x1_.size(0), -1)

        x1_c = self.classifier1(x1_f)

        x2_ = self.conv_block2(x2)
        x2_ = self.max_pool2(x2_)
        x2_f = x2_.view(x2_.size(0), -1)
        x2_c = self.classifier2(x2_f)

        x3_ = self.conv_block3(x3)
        x3_ = self.max_pool3(x3_)
        x3_f = x3_.view(x3_.size(0), -1)
        x3_c = self.classifier3(x3_f)

        x_c_all = torch.cat((x1_f, x2_f, x3_f), -1)
        x_c_all = self.classifier_concat(x_c_all)

        return x1_c, x2_c, x3_c, x_c_all

def cosine_anneal_schedule(t, T, lr):
    cos_inner = np.pi * (t % T)
    cos_inner /= T
    cos_out = np.cos(cos_inner) + 1
    return float(lr / 2 * cos_out)

def train(nb_epoch, batch_size, store_name, resume=False, start_epoch=0, model_path=None):
    exp_dir = store_name
    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)

    use_cuda = torch.cuda.is_available()
    print(f"CUDA available: {use_cuda}")

    print('==> Preparing data..')
    # 修改数据预处理
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),  # 使用Resize代替Scale
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 使用测试数据集
    trainset = torchvision.datasets.ImageFolder(root='Image-based Recognition/dataset_test/train',
                                                transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    print(f"Dataset size: {len(trainset)} images")
    print(f"Number of batches: {len(trainloader)}")

    # 创建模型
    model_name = "skresnext50_32x4d"
    net = timm.create_model(model_name, pretrained=True, num_classes=10).cuda()

    net_layers = list(net.children())
    net_layers = net_layers[0:8]

    net = Network_Wrapper(net_layers)

    print('Model %s created, param count: %d' %
          ('Created_model', sum([m.numel() for m in net.parameters()])))

    # 设置并行和设备
    if use_cuda:
        netp = torch.nn.DataParallel(net).cuda()
        device = torch.device("cuda")
    else:
        netp = net
        device = torch.device("cpu")

    net.to(device)
    cudnn.benchmark = True

    # 设置损失函数和优化器
    CELoss = nn.CrossEntropyLoss()
    optimizer = optim.SGD([
        {'params': net.classifier_concat.parameters(), 'lr': 0.002},
        {'params': net.conv_block1.parameters(), 'lr': 0.002},
        {'params': net.classifier1.parameters(), 'lr': 0.002},
        {'params': net.conv_block2.parameters(), 'lr': 0.002},
        {'params': net.classifier2.parameters(), 'lr': 0.002},
        {'params': net.conv_block3.parameters(), 'lr': 0.002},
        {'params': net.classifier3.parameters(), 'lr': 0.002},
        {'params': net.Features.parameters(), 'lr': 0.0002}
    ],
        momentum=0.9, weight_decay=5e-4)

    # 训练循环
    max_val_acc = 0
    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]

    for epoch in range(start_epoch, nb_epoch):
        net.train()
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        train_loss4 = 0
        correct = 0
        total = 0
        idx = 0

        # 使用 tqdm 显示训练进度条
        pbar = tqdm(enumerate(trainloader), total=len(trainloader))
        for batch_idx, (inputs, targets) in pbar:
            # 跳过不完整的批次
            if inputs.shape[0] < batch_size:
                continue

            idx = batch_idx

            # 应用数据增强
            try:
                inputs1 = inputs.clone()
                inputs2 = inputs.clone()
                inputs3 = inputs.clone()

                # 应用渐进式增强
                inputs1 = img_progressive(inputs1, 0.3, p=0.3)
                inputs2 = img_progressive(inputs2, 0.2, p=0.3)
                inputs3 = img_progressive(inputs3, 0.1, p=0.3)

                # 移动到设备
                if use_cuda:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    inputs1 = inputs1.to(device)
                    inputs2 = inputs2.to(device)
                    inputs3 = inputs3.to(device)

                # 更新学习率
                for nlr in range(len(optimizer.param_groups)):
                    optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, lr[nlr])

                # 训练第一个分类器
                optimizer.zero_grad()
                output_1, _, _, _ = netp(inputs1)
                loss1 = CELoss(output_1, targets) * 1
                loss1.backward()
                optimizer.step()

                # 训练第二个分类器
                optimizer.zero_grad()
                _, output_2, _, _ = netp(inputs2)
                loss2 = CELoss(output_2, targets) * 1
                loss2.backward()
                optimizer.step()

                # 训练第三个分类器
                optimizer.zero_grad()
                _, _, output_3, _ = netp(inputs3)
                loss3 = CELoss(output_3, targets) * 1
                loss3.backward()
                optimizer.step()

                # 训练组合分类器
                optimizer.zero_grad()
                _, _, _, output_concat = netp(inputs)
                concat_loss = CELoss(output_concat, targets) * 2
                concat_loss.backward()
                optimizer.step()

                # 计算准确率
                _, predicted = torch.max(output_concat.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

                # 累计损失
                train_loss += (loss1.item() + loss2.item() + loss3.item() + concat_loss.item())
                train_loss1 += loss1.item()
                train_loss2 += loss2.item()
                train_loss3 += loss3.item()
                train_loss4 += concat_loss.item()

                # 更新进度条信息
                pbar.set_description(f'Epoch {epoch}: Loss: {train_loss / (batch_idx + 1):.3f}, Acc: {100. * float(correct) / total:.3f}%')

            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # 保存训练结果
        if idx > 0:  # 确保至少处理了一个批次
            train_acc = 100. * float(correct) / total
            train_loss = train_loss / (idx + 1)
            with open(exp_dir + '/results_train.txt', 'a') as file:
                file.write(
                    'Iteration %d | train_acc = %.5f | train_loss = %.5f | Loss1: %.3f | Loss2: %.5f | Loss3: %.5f | Loss_concat: %.5f |\n' % (
                        epoch, train_acc, train_loss, train_loss1 / (idx + 1), train_loss2 / (idx + 1), train_loss3 / (idx + 1),
                        train_loss4 / (idx + 1)))

        # 保存模型
        net.cpu()
        torch.save(net, './' + store_name + '/model.pth')
        if use_cuda:
            net.to(device)

        # 打印每个 epoch 的参数情况
        print(f"Epoch {epoch} completed. Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}%")

if __name__ == '__main__':
    save_path = 'save_teacher_test'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    train(nb_epoch=10,        
          batch_size=8,      #
          store_name=save_path,
          resume=False,
          start_epoch=0,
          model_path='')
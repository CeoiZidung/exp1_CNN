# -*- coding:utf-8 -*-  
'''Train CIFAR10 with PyTorch.'''

from __future__ import print_function
#防止爬虫爬取数据出现错误
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from utils import progress_bar  #进度条函数
import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter  #引入结果可视化模型


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

# 定义是否使用GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 超参数设置
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
train_loss_everyepoch = 0
train_accuracy_everyepoch = 0
test_loss_everyepoch = 0
test_accuracy_everyepoch = 0


# Data
print('==> Preparing data..')
# torchvision.transforms是pytorch中的图像预处理包 一般用Compose把多个步骤整合到一起：
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),   # 先四周填充0，再把图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),      # 以0.5的概率水平翻转给定的PIL图像
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


# DataLoader接口主要用来将自定义的数据读取接口的输出或者PyTorch已有的数据读取接口的输入按照batch size封装成Tensor
# 生成一个个batch进行批训练，组成batch的时候顺序打乱取
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)

print('SAMPLES:', len(trainset), len(testset))
print('EPOCH:', len(trainloader), len(testloader))
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck') # 类型

# Model
print('==> Building model..')

# 定义卷积神经网络
class DaiNet(nn.Module): # 我们定义网络时一般是继承的torch.nn.Module创建新的子类
    def __init__(self):
        super(DaiNet, self).__init__()      # 第二、三行都是python类继承的基本操作,此写法应该是python2.7的继承格式,但python3里写这个好像也可以
        self.conv1 = nn.Conv2d(3, 12, 5)    # 添加第一个卷积层,调用了nn里面的Conv2d
                                            # 输入是3通道的图像，输出是12通道，也就是12个卷积核，卷积核是5*5，其余参数都是用的默认值
        self.dp = nn.Dropout(0.5)
        self.pool = nn.MaxPool2d(2, 2)      # 最大池化层  MaxPool2d(窗口大小, 窗口移动的步长, padding=0, 一个控制窗口中元素步幅的参数=1, )
        
        self.conv2 = nn.Conv2d(12, 24, 3)    # 同样是卷积层 
        
        self.dp = nn.Dropout(0.5)
        #self.conv3 = nn.Conv2d(24, 32, 3)    # 与baseline对比，去掉第三层

        self.fc1 = nn.Linear(24 * 6 * 6, 120)   # 接着三个全连接层 Linear(in_features, out_features, bias=True)
        self.fc2 = nn.Linear(120, 84)          
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))    # F是torch.nn.functional的别名，这里调用了relu函数 F.relu()
        x = self.pool(F.relu(self.conv2(x)))
        
        # fully connect
        x = x.view(-1, 24 * 6 * 6)  
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
net = DaiNet()
netname = 'DaiNet'
writer_train = SummaryWriter(comment='DaiNet_train') # 提供一个 comment 参数，将使用 runs/日期时间-comment 路径来保存日志
writer_test = SummaryWriter(comment='DaiNet_test')

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join('./checkpoint', netname, 'ckpt.t7'))
    #checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# 定义损失函数和优化器 
criterion = nn.CrossEntropyLoss()   # 损失函数为交叉熵，多用于多分类问题
# 优化方式为mini-batch momentum-SGD  SGD梯度优化方式---随机梯度下降 ，并采用L2正则化（权重衰减）
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Training 训练网络
def train(epoch):
    print('\nTrain Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device) 
        optimizer.zero_grad()   # 梯度清零 zero the parameter gradients
        outputs = net(inputs)   # forward
        loss = criterion(outputs, targets)  # loss 计算损失值,criterion我们在第三步里面定义了
        loss.backward()     # 执行反向传播 backward  就是在实现反向传播，自动计算所有的梯度
        optimizer.step()    # 更新参数 update weights 当执行反向传播之后，把优化器的参数进行更新，以便进行下一轮
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    train_loss_everyepoch = train_loss/(batch_idx+1)
    train_accuracy_everyepoch = 100.*correct/total
    return (train_loss_everyepoch,train_accuracy_everyepoch)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        test_error_everyepoch = test_loss/(batch_idx+1)
        test_accuracy_everyepoch = 100.*correct/total

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if not os.path.isdir('checkpoint/'+netname):
            os.mkdir('checkpoint/'+netname)
        torch.save(state, './checkpoint/'+netname + '/ckpt.t7')
        best_acc = acc
    return (test_error_everyepoch,test_accuracy_everyepoch)

# 训练网络
for epoch in range(start_epoch, start_epoch+200):
    (train_loss_everyepoch,train_accuracy_everyepoch) = train(epoch)
    (test_loss_everyepoch,test_accuracy_everyepoch) = test(epoch)
    #tensorboard 记入数据进行可视化
    writer_train.add_scalar('loss', train_loss_everyepoch, global_step= epoch)
    writer_train.add_scalar('accuracy', train_accuracy_everyepoch, global_step= epoch)
    writer_test.add_scalar('loss', test_loss_everyepoch, global_step= epoch)
    writer_test.add_scalar('accuracy', test_accuracy_everyepoch, global_step= epoch)
    

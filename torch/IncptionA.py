import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

batch_size = 64

# 图像处理,标准化
# 由于标准化处理之后便于矩阵运算，有利于提高运算速率
# 单通道实现为多通道 28*28 -> 1*28*28, 同时进行标准化
# 0.1307， 0.3081都是均值与方差
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])

# 加载训练数据集
train_dataset = datasets.MNIST(root='../dataset/mnist/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

# 加载测试数据集
test_dataset = datasets.MNIST(root='../dataset/mnist/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

class IncptionA(torch.nn.Module):
    def __init__(self, in_channels):
        super(IncptionA, self).__init__()
        self.branch1x1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=1)

        self.branch5x5_1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=1)
        self.branch5x5_2 = torch.nn.Conv2d(in_channels=16, out_channels=24, kernel_size=5, padding=2)

        self.branch3x3_1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=1)
        self.branch3x3_2 = torch.nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, padding=1)
        self.branch3x3_3 = torch.nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, padding=1)

        self.branchpool = torch.nn.Conv2d(in_channels=in_channels, out_channels=24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        # 这里存在有问题，不理解为什么会有卷积运算
        branchpool = F.avg_pool2d(x, kernel_size=3, padding=1, stride=1)
        branchpool = self.branchpool(branchpool)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        output = [branch1x1, branch5x5, branch3x3, branchpool]
        # 这里 dim是什么意思？->指的是b c w h，dim=1指的就是按channel进行拼接
        return torch.cat(output, dim=1)

class NET(torch.nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(88, 20, kernel_size=5)

        self.incep1 = IncptionA(in_channels=10)
        self.incep2 = IncptionA(in_channels=20)

        self.mp = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(1408, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)

        return x

model = NET()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
# 定义损失函数
criterion = torch.nn.CrossEntropyLoss()
# 定义优化器
# momentum 是指的是冲量值，用来优化神经网络
optimization = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# 定义超参数

# 定义训练过程
def train(epoch):
    # 设置模型为训练模式
    model.train()
    running_loss = 0.0
    for batch_index, data in enumerate(train_loader, 0):
        inputs, labels = data
        # 将输入和标签转移到相应设备（CPU或GPU）
        inputs, labels = inputs.to(device), labels.to(device)
        # 优化器初始化
        optimization.zero_grad()
        # 前向传播
        output = model(inputs)
        # 计算损失
        loss = criterion(output, labels)
        # 反向传播
        loss.backward()
        # 更新参数
        optimization.step()
        # 累加损失
        running_loss += loss.item()

        # 每300个批次打印一次平均损失
        if batch_index % 300 == 299:
            print('[%d, %d] loss = %f' % (epoch + 1, batch_index + 1, running_loss / 300))
            running_loss = 0.0

# 定义测试过程
def test():
    # 设置模型为评估模式
    model.eval()
    correct = 0
    total = 0
    # 关闭梯度计算
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            # 将输入和标签转移到相应设备（CPU或GPU）
            inputs, labels = inputs.to(device), labels.to(device)
            # 前向传播
            outputs = model(inputs)
            # 获取预测结果
            _, predicted = torch.max(outputs.data, 1)
            # 由于是N * 1矩阵，那么size就是（N，1）的元组，我们可以直接选第一个元素表示这一个批次的数量
            total += labels.size(0)
            # 计算正确预测的数量
            correct += (predicted == labels).sum().item()
    # 打印准确率
    print('Accuracy is %f%%' % ((correct / total) * 100))

if __name__=='__main__':
    for epoch in range(100):
        train(epoch)
        test()

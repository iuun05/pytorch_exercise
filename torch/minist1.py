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


class NET(torch.nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.pooling((self.conv1(x))))
        x = F.relu(self.pooling((self.conv2(x))))
        x = x.view(batch_size, -1)
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

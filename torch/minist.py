import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score

import torch.optim as optim

# 定义超参数
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

# 定义神经网络结构
class NET(torch.nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        self.n1 = torch.nn.Linear(784, 512)
        self.n2 = torch.nn.Linear(512, 256)
        self.n3 = torch.nn.Linear(256, 128)
        self.n4 = torch.nn.Linear(128, 64)
        self.n5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        # 展平输入图像
        x = x.view(-1, 784)
        # 依次通过每一层，并使用ReLU激活函数
        x = F.relu(self.n1(x))
        x = F.relu(self.n2(x))
        x = F.relu(self.n3(x))
        x = F.relu(self.n4(x))
        # 最后一层不使用激活函数，输出分类结果
        return self.n5(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NET().to(device)
criterion = torch.nn.CrossEntropyLoss()
# momentum 是指的是冲量值，用来优化神经网络
optimization = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

def train(epoch):
    # 设置模型为训练模式
    model.train()
    running_loss = 0.0
    for batch_index, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimization.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimization.step()
        running_loss += loss.item()

        if batch_index % 300 == 299:
            print('[%d, %d] loss = %f' % (epoch + 1, batch_index + 1, running_loss / 300))
            running_loss = 0.0

def test():
    # 设置模型为评估模式
    model.eval()
    correct = 0
    total = 0

    all_labels = []
    all_preds = []
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

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            total += labels.size(0)
            # 计算正确预测的数量
            correct += (predicted == labels).sum().item()

    # 打印准确率
    accuracy = accuracy_score(all_labels, all_preds)
    print('Accuracy is %f%%' % (accuracy * 100))

    # 精准率、召回率、F1值
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print('Precision is %f%%' % (precision * 100))
    print('Recall is %f%%' % (recall * 100))
    print('F1 is %f%%' % (f1 * 100))

    # 计算混淆矩阵
    confu_matr = confusion_matrix(all_labels, all_preds)
    print('Confusion Matrix is \n', confu_matr)

if __name__ == '__main__':
    for epoch in range(20):
        # 训练模型
        train(epoch)
        # 测试模型
    print('=====Finished Training=====')
    test()


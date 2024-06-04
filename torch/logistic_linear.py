import torch
import torch.nn.functional as F

# 标准模板
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        # 调用父类的super
        super(LogisticRegressionModel, self).__init__()
        # 构造对象，包含权重
        # 输入的样本的大小，输出样本的大小，以及是否存在偏置(默认是True)
        self.linear = torch.nn.Linear(1, 1)

    # 此时的forward已经是override
    def forward(self, x):
        # 可调用的对象，其中存在有神奇的__call__函数，导致调用可以直接调用call函数
        y_pred = F.sigmoid(self.linear(x))

        return y_pred


x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])
model = LogisticRegressionModel()
# loss函数
criterion = torch.nn.BCELoss(reduction='sum')
# 随机梯度下降优化器，对应相关的参数，以及学习率
# 优化器不会构建运算图，相关其他参数参考文档
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('W = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

x_test = torch.Tensor([[4.0]])
y_test = model(x_test)

print('y_test = ', y_test)

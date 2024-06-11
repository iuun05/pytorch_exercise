import numpy as np
import torch

# 此处是根据数据集的分配方式的方法来处理
xy = np.loadtxt('D:\mywork\exp01\Data\diabetes.csv.gz', delimiter=',', dtype=np.float32)

x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()
        # 改用Relu需要注意的是，我们需要将值保证在0－1之间，因此我们需要做归一化
        # self.activate = torch.nn.ReLU()
    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        # x = self.activate(self.linear1(x))
        # x = self.activate(self.linear2(x))
        # x = self.activate(self.linear3(x))
        return x

model = Model()

criterion = torch.nn.BCELoss(reduction='sum')
optimization = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(1000):
    y_pred = model.forward(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimization.zero_grad()
    loss.backward()
    optimization.step()


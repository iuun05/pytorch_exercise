import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
class diabetesDataSet(Dataset):
    def __init__(self, filepath='D:\mywork\exp01\Data\diabetes.csv.gz'):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

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

if __name__ == '__main__':
    dataset = diabetesDataSet()
    model = Model()
    criterion = torch.nn.BCELoss(reduction='sum')
    optimization = torch.optim.SGD(model.parameters(), lr=0.01)

    train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=4)
    for epoch in range(1000):
        for i, data in enumerate(train_loader, 0):

            inputs, labels = data

            y_pred = model.forward(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, loss.item())

            optimization.zero_grad()
            loss.backward()
            optimization.step()
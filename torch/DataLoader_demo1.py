import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# DataSet 是一个抽象类，只能用于继承不能用于实例化
class diabetesDataSet(Dataset):
    def __init__(self):
        pass

    # 随机读取数据集
    def __getitem__(self, item):
        pass
    # 得到数据长度
    def __len__(self):
        pass


dataset = Dataset()
train_loade = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=4)
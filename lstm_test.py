import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from torch import nn
from torch.autograd import Variable

data_csv = pd.read_csv('F:/data_test/data.csv', usecols=[1])
plt.plot(data_csv)

plt.show()

# 数据预处理
data_csv = data_csv.dropna()  # 删除缺失数据
dataset = data_csv.values
dataset = dataset.astype('float32')  # 转换数据类型
max_value = np.max(dataset)
min_value = np.min(dataset)
scalar = max_value - min_value
dataset = list(map(lambda x: x / scalar, dataset))  # 数据归一化?


# 用前两个月的流量预测当月流量 所以look_back=2
def create_dataset(dataset, look_back=2):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]  # 选两个 当前的和下一个
        dataX.append(a)
        dataY.append(dataset[i + look_back])  # 从第三个数据开始
    return np.array(dataX), np.array(dataY)


# 创建好输入输出
data_X, data_Y = create_dataset(dataset)

# 划分训练集和测试集，70% 作为训练集
train_size = int(len(data_X) * 0.7)
test_size = len(data_X) - train_size
train_X = data_X[:train_size]
train_Y = data_Y[:train_size]
test_X = data_X[train_size:]
test_Y = data_Y[train_size:]

train_X = train_X.reshape(-1, 1, 2)  # 后两维固定为1,2，第一维自动确定
train_Y = train_Y.reshape(-1, 1, 1)
test_X = test_X.reshape(-1, 1, 2)

train_x = torch.from_numpy(train_X)  # 转为tensor形式
train_y = torch.from_numpy(train_Y)
test_x = torch.from_numpy(test_X)


# print('trainx{}\ntrainy{}\ntestx{}'.format(train_x,train_y,test_x))

# 定义模型
class lstm_reg(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super(lstm_reg, self).__init__()  # 调用父类(超类)的一个方法

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)  # rnn
        self.reg = nn.Linear(hidden_size, output_size)  # 回归

    def forward(self, x):
        x, _ = self.rnn(x)  # (seq, batch, hidden)
        s, b, h = x.shape
        x = x.view(s * b, h)  # view改变维度；转换成线性层的输入格式，因为nn.Linear 不接受三维的输入，所以我们先将前两维合并在一起
        x = self.reg(x)  # 经过线性层
        x = x.view(s, b, -1)  # 将前两维分开，最后输出结果
        return x


net = lstm_reg(2, 4)

criterion = nn.MSELoss()  # 均方根损失函数
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)  # 优化步骤

# 开始训练
for e in range(1000):
    var_x = Variable(train_x)
    var_y = Variable(train_y)
    # 前向传播
    out = net(var_x)
    loss = criterion(out, var_y)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (e + 1) % 100 == 0:  # 每 100 次输出结果
        print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.item()))

net = net.eval()  # 转换成测试模式

data_X = data_X.reshape(-1, 1, 2)
data_X = torch.from_numpy(data_X)
var_data = Variable(data_X)
pred_test = net(var_data)  # 测试集的预测结果

# 改变输出的格式
pred_test = pred_test.view(-1).data.numpy()

# 画出实际结果和预测的结果
plt.plot(pred_test, 'r', label='prediction')
plt.plot(dataset, 'b', label='real')
plt.legend(loc='best')
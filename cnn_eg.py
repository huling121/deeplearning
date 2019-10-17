import torch.nn as nn
import torch as t
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)

        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(x.size()[0],-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
input = Variable(t.randn(1,1,32,32))
net = Net()
#print('net.conv1.bias.grad')
#print(net.conv1.bias.grad)
weight = 0
learning_rate = 0.01
# weight = weight - learning_rate*gradient
optimizer = optim.SGD(net.parameters(),lr=0.01)
optimizer.zero_grad()
output = net(input)
# print(out.size())
# print(net)
target = Variable(t.arange(0,10)) #(生成 ([0,1,2,3,4,5,6,7,8,9]))
target = target.float()
target = target.view([1,10])

criterion = nn.MSELoss()
loss = criterion(output,target)
net.zero_grad()
loss.backward()
for var in net.parameters():
    var.data.sub_(var.grad.data*learning_rate)
optimizer.step()
print(loss)



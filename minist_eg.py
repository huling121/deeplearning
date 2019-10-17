import torch
import torchvision
from torch.autograd import Variable
import torch.utils.data.dataloader as Data
import torchsnooper
train_data = torchvision.datasets.MNIST(
    './mnist',train = True,transform=torchvision.transforms.ToTensor(),download=True
)
test_data = torchvision.datasets.MNIST(
    './mnist',train=False,transform=torchvision.transforms.ToTensor()
)
print('train_data:',train_data.train_data.size())
print('train_labels:',train_data.train_labels.size())
#print(train_data.train_labels)
print('test_data:',test_data.test_data.size())
print(test_data.test_labels)
train_loader = Data.DataLoader(dataset=train_data,batch_size=64,shuffle=True)
test_loader = Data.DataLoader(dataset=test_data,batch_size=64)
class Net(torch.nn.Module):
    #@torchsnooper.snoop()
    def __init__(self,args):
        super(Net,self).__init__()
        self.args = args
        self.use_cuda = args['use_cuda']
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1,32,3,1,1),#（1，32，kernel_size=(3,3),stride = (1,1),padding=(1,1)）
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)) #(kernel_size =2,stide=2,padding=0,dilition=1)
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32,64,3,1,1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64,64,3,1,1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(64*3*3,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,10)
        )

    def forward(self,x):  # x的大小是  （64，1，28，28）
        conv1_out = self.conv1(x) # 卷积一次之后的大小：(64,32,14,14)
        conv2_out = self.conv2(conv1_out) # 卷积2后的大小:(64,64,7,7)
        conv3_out = self.conv3(conv2_out) # 卷积3后的大小:(64,64,3,3)
        res = conv3_out.view(conv3_out.size(0),-1) # 调整之后的大小：（64，576）
        out = self.dense(res) # out的值： （64，10）
        return out
args = {}
args['use_cuda'] = True
model = Net(args)
if args['use_cuda']:
    model = model.cuda()
#print(model)
optimizer = torch.optim.Adam(model.parameters())# 优化器
loss_func = torch.nn.CrossEntropyLoss() # 损失函数
for epoch in range(10):
    print('epoch {}'.format(epoch + 1))
    # training------------------
    train_loss = 0.
    train_acc = 0.
    for batch_x,batch_y in train_loader:
        batch_x,batch_y = Variable(batch_x).cuda(),Variable(batch_y).cuda()
        out = model(batch_x)
        loss =loss_func(out,batch_y)
        train_loss += loss.item()
        pred = torch.max(out,1)[1]
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Train Loss: {: .6f},Acc:{: .6f}'.format(train_loss/(len(train_data)),train_acc/(len(train_data))))

    # evaluate------------------
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    for batch_x,batch_y in test_loader:
        batch_x,batch_y = Variable(batch_x).cuda(),Variable(batch_y).cuda()
        out = model(batch_x)
        print('测试之后的值：',out)
        print('验证值：',batch_y)
        loss = loss_func(out,batch_y)
        eval_loss += loss.item()
        pred = torch.max(out,1)[1]
        num_correct = (pred == batch_y).sum()
        eval_acc += num_correct.item()
    print('Test Loss: {: .6f},Acc :{: .6f}'.format(eval_loss/(len(test_data)),eval_acc/(len(test_data))))






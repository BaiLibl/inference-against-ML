import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms 
from torch.autograd import Variable
#transforms用来对Imgae做处理，比如转成tensor，做数据增强

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='./runs/')
# tensorboard --logdir=runs --port 8123

# network
class net_CNN(nn.Module):
        def __init__(self):
            super(net_CNN, self).__init__()
            self.covn1 = nn.Conv2d(1, 10, kernel_size=5) #输入是1通道，用10个5*5的kernel
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.pooling = nn.MaxPool2d(2)
            self.fc = nn.Linear(320, 10)

        def forward(self, x):
            in_size = x.size(0)
            x = F.relu(self.pooling(self.covn1(x)))
            x = F.relu(self.pooling(self.conv2(x)))
            x = x.view(in_size, -1) #flatten
            x = self.fc(x)
            output = F.log_softmax(x)
            return output

# data
batch_size = 64
train_dataset = datasets.MNIST(root='./data/',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)
test_dataset = datasets.MNIST(root='./data/',
                                train=False,
                                transform=transforms.ToTensor(),
                                download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)


model = net_CNN()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = torch.nn.NLLLoss()
model_save = "./save/"
# pipeline
def train(epoch):
    for i, (data, label) in enumerate(train_loader):
        data, label = Variable(data), Variable(label)
        optimizer.zero_grad()
        output = model(data)
        # loss = F.nll_loss(output, label)
        loss = criterion(output.squeeze(), label)
        loss.backward()
        optimizer.step()
        if i % 200 == 0:
            print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
    
    writer.add_scalar('Loss/train', loss / len(train_loader.dataset), epoch)


def test(epoch):
    loss = 0
    acc = 0
    for i, (data, label) in enumerate(test_loader):
        output = model(data)
        # loss += F.nll_loss(output, label, size_average=False).data[0]
        loss += criterion(output, label).sum()
        pred = output.data.max(1, keepdim=True)[1]
        acc += pred.eq(label.data.view_as(pred)).cpu().sum()
    
    loss = loss / len(test_loader.dataset)
    acc = acc / len(test_loader.dataset)
    print('Test loss: {}, acc:{}'.format(loss, acc))
    writer.add_scalar('Loss/test', loss / len(test_loader.dataset), epoch)
    writer.add_scalar('Acc/test', acc, epoch)

def run():

    for epoch in range(1, 2):
        train(epoch)
        test(epoch)
    import time
    timestamp = time.time()
    torch.save(model, model_save+"/"+str(timestamp)) 
    

if __name__ == '__main__':
    run()
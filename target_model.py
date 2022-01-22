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
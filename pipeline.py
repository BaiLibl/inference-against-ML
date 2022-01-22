import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms 
from torch.autograd import Variable

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='./runs/')
# tensorboard --logdir=runs --port 8123

import datas
import target_model

batch_size = 64
criterion = torch.nn.NLLLoss()
model_save = "./save/"
dname = 'MNIST'

# load data
target_train, target_test, shadow_train, shadow_test = datas.load_data(dname)
target_train_loader = torch.utils.data.DataLoader(target_train, batch_size=batch_size, shuffle=True)
target_test_loader  = torch.utils.data.DataLoader(target_test,  batch_size=batch_size, shuffle=False)
shadow_train_loader = torch.utils.data.DataLoader(shadow_train, batch_size=batch_size, shuffle=True)
shadow_test_loader  = torch.utils.data.DataLoader(shadow_test,  batch_size=batch_size, shuffle=False)

# target model
model = target_model.net_CNN()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)



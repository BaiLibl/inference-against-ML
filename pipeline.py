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

model_save = "./save/"
# pipeline
def one_epoch(epoch, train_loader, test_loader):
    # train
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

    # test
    loss = 0
    acc = 0
    for i, (data, label) in enumerate(test_loader):
        output = model(data)
        print(output[0])
        # loss += F.nll_loss(output, label, size_average=False).data[0]
        loss += criterion(output, label).sum()
        pred = output.data.max(1, keepdim=True)[1]
        acc += pred.eq(label.data.view_as(pred)).cpu().sum()
    
    loss = loss / len(test_loader.dataset)
    acc = acc / len(test_loader.dataset)
    print('Test loss: {}, acc:{}'.format(loss, acc))
    writer.add_scalar('Loss/test', loss / len(test_loader.dataset), epoch)
    writer.add_scalar('Acc/test', acc, epoch)

def run(train_loader, test_loader):

    for epoch in range(1, 2):
        one_epoch(epoch, train_loader, test_loader)
    import time
    timestamp = time.time()
    torch.save(model, model_save+"/epoch"+str(epoch)) 
    

if __name__ == '__main__':
    run(target_train_loader, target_test_loader)

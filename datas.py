import torch 
import torch.utils.data as data_utils


from torchvision import datasets, transforms 
import numpy as np
# from torch.autograd import Variable 

# load_data: target train/test, shadow train/test
def load_data(dname, r=0.5):
    if dname == "MNIST": 
        # train:6w, test:1w
        batch_size = 64
        train_dataset = datasets.MNIST(root='./data/',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)
        test_dataset = datasets.MNIST(root='./data/',
                                train=False,
                                transform=transforms.ToTensor(),
                                download=True)
        data = train_dataset + test_dataset
        t_len = int(r * len(data))
        s_len = len(data) - t_len
        print(dname, len(train_dataset), len(test_dataset))
        
        indices = torch.arange(t_len)
        target_data = data_utils.Subset(data, indices)
        indices = torch.arange(int(t_len/2))
        target_train = data_utils.Subset(target_data, indices)
        indices = torch.arange(int(t_len/2),t_len)
        target_test = data_utils.Subset(target_data, indices)

        # non-member/member = 1
        indices = torch.arange(s_len, len(data))
        shadow_data = data_utils.Subset(data, indices)
        indices = torch.arange(int(s_len/2))
        shadow_train = data_utils.Subset(shadow_data, indices)
        indices = torch.arange(int(s_len/2),s_len)
        shadow_test = data_utils.Subset(shadow_data, indices)

        print("%s: target_train_size:%d, traget_test_size:%d, shadow_train_size:%d, shadow_test_size:%d" % (dname, len(target_train), len(target_test), len(shadow_train), len(shadow_test)))
        return target_train, target_test, shadow_train, shadow_test 
        # target_train_loader = torch.utils.data.DataLoader(target_train, batch_size=batch_size, shuffle=True)
        # target_test_loader  = torch.utils.data.DataLoader(target_test,  batch_size=batch_size, shuffle=False)
        # shadow_train_loader = torch.utils.data.DataLoader(shadow_train, batch_size=batch_size, shuffle=True)
        # shadow_test_loader  = torch.utils.data.DataLoader(shadow_test,  batch_size=batch_size, shuffle=False)

        # return target_train_loader, target_test_loader, shadow_train_loader, shadow_test_loader


# load_data('MNIST')

# get_data
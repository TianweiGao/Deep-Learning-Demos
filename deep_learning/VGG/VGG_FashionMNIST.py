import torch
import torch.nn as nn
import torch.functional as F
from torch.optim import SGD,Adagrad,adam
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm  
import pdb
from tools import *

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(20,256)
        self.linear2 = nn.Linear(256,10)

    def forward(self,x):
        for layer in self._modules.values():
            x = layer(x)
        return x
### Load FashionMNIST dataset

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()  # Converts PIL image or numpy array to PyTorch tensor
])

train_data = datasets.FashionMNIST(
    root = './FashionMNIST',
    train = True,
    download = True,
    transform = transform,
)

test_data = datasets.FashionMNIST(
    root = './FashionMNIST',
    train = False,
    download = True,
    transform = transform,
)

# Define the split ratio
train_ratio = 0.8  # 80% for training
val_ratio = 0.2    # 20% for validation

train_size = int(train_ratio * len(train_data))
val_size = len(train_data) - train_size

train_subset, valid_subset = random_split(train_data, [train_size, val_size])

train_dataloader = DataLoader(train_subset,batch_size=64,shuffle=True)
valid_dataloader = DataLoader(valid_subset,batch_size=64,shuffle=True)
test_dataloader = DataLoader(test_data,batch_size=64,shuffle=True)

print(f"Train DataLoader has batch number: {len(train_dataloader)}")
### Build VGG Blocks
class vgg_blocks(nn.Module):
    def __init__(self,num_convs,out_channels):
        super().__init__()
        for i in range(num_convs):
            
            self.add_module(f'conv{i}',nn.LazyConv2d(out_channels,kernel_size=3,padding=1))
            self.add_module(f'relu{i}',nn.ReLU())
            
        self.add_module('pooling', nn.MaxPool2d(kernel_size=2,stride=2))

    def forward(self,x):
        for module in self._modules.values():
            x = module(x)
        return x
class VGG(nn.Module):
    def __init__(self,arch,num_classes = 10):
        super().__init__()
        for i, (num_convs,out_channels) in enumerate(arch):
            self.add_module(f'vgg_block{i}',vgg_blocks(num_convs,out_channels))    
        self.add_module(f'Linear',nn.Sequential(nn.Flatten(),nn.LazyLinear(4096),nn.ReLU(),nn.Dropout(0.5),nn.LazyLinear(4096),nn.ReLU(),nn.Dropout(0.5),nn.LazyLinear(num_classes)))

    def forward(self,x):
        for module in self._modules.values():
            x = module(x)
        return x

if __name__=="__main__":

    model = VGG(arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)))

    def gpu(i=0): 
        return torch.device(f'cuda:{i}')

    device = gpu()

    model.to(device)

    optimizer = optim.Adam(model.parameters(),lr=1e-3)

    train_and_valid(model,train_dataloader,valid_dataloader,criterion=nn.CrossEntropyLoss(),optimizer=optimizer,
                    output_cache='./',device=gpu(),file_name='VGG_FashionMNIST',
                    max_epochs=6)

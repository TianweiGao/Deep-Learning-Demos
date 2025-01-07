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
from VGG_FashionMNIST import *
device = gpu()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()  # Converts PIL image or numpy array to PyTorch tensor
])
test_data = datasets.FashionMNIST(
    root = './FashionMNIST',
    train = False,
    download = True,
    transform = transform,
)
test_dataloader = DataLoader(test_data,batch_size=64,shuffle=True)

model = torch.load('/home/deep_learning/VGG/VGG_FashionMNIST.pth')
model.eval()
simpleClassifierTest(model,test_dataloader,device)
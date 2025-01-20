#!/usr/bin/env python
# coding: utf-8

# In[32]:


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
from torch.utils.data import DataLoader
from tqdm import tqdm  
import pdb
from tools import *
from ViT.ViT_FashionMNIST import *
from VGG.VGG_FashionMNIST import *

def train_and_valid_with_teacher(model,teacher,train_dataloader,valid_dataloader,criterion,teaching_criterion,optimizer,
                    output_cache,device,file_name,batch_tqdm=False,
                    max_epochs=20,):
    train_loss_list = []
    test_loss_list = []
    for epoch in tqdm(range(max_epochs),desc='Epochs',):
        loss = 0.
        
        if epoch==0 and batch_tqdm:
            train_iter = tqdm(train_dataloader,desc = 'Batch',)
        else:
            train_iter = train_dataloader
            #print('tqdm_adjusted')
        for batch, (X,y) in enumerate(train_iter):
            
            y_pred = model(X.to(device))
            y_teach = teacher(X.to(device)) 
            batch_loss = criterion(y_pred,y.to(device)) + teaching_criterion(y_pred,y_teach)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            loss += criterion(y_pred,y.to(device)).detach().cpu().numpy()
            
        train_loss_list.append(loss) 

        loss = 0.
        for batch, (X,y) in enumerate(valid_dataloader):
            y_pred = model(X.to(device))
            batch_loss = criterion(y_pred,y.to(device))
            loss += batch_loss.detach().cpu().numpy()
        test_loss_list.append(loss)

        if epoch%2==0:
            print(f"Epoch: {epoch+1}, Train Loss: {np.round(train_loss_list[-1],3)}, Test Loss: {np.round(test_loss_list[-1],3)}")
    
    fp = output_cache
    torch.save(model,f'{fp+"/"+file_name}.pth')
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_epochs + 1), train_loss_list, label='Train Loss', marker='o')
    plt.plot(range(1, max_epochs + 1), test_loss_list, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.savefig(f'{fp}/Loss of {file_name}.png') 




if __name__=="__main__":
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


    device = gpu()

    img_size, patch_size = 224, 16
    num_hiddens, mlp_num_hiddens, num_heads, num_blks = 512, 2048, 8, 2
    emb_dropout, blk_dropout, lr = 0.1, 0.1, 0.1
    model = ViT(img_size, patch_size, num_hiddens, mlp_num_hiddens, num_heads,
                num_blks, emb_dropout, blk_dropout, lr)

    teacher = torch.load('./models/VGG_FashionMNIST.pth')

    model.to(device)
    optimizer = optim.SGD(model.parameters(),lr=1e-2)
    teacher.to(device)
    def teaching_criterion(y_pred,y_true,T=5):
        return -(torch.softmax(y_true,dim=1)*y_pred).sum()


    trainer = train_and_valid_with_teacher(model,teacher,train_dataloader,valid_dataloader,criterion=nn.CrossEntropyLoss(),teaching_criterion=teaching_criterion,optimizer=optimizer,
                            output_cache = '~/Deep-Learning-Demos/deep_learning/DeiT', device = device,file_name='DeiT_FashionMNIST', batch_tqdm=False,
                            max_epochs=1)








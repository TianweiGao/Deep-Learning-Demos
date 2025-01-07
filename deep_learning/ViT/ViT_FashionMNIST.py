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


# In[24]:


class ScaledDotProduction(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    # shape of query: (batch_size, no. of query, feature_dim)
    # shape of keys: (batch_size, no. of key-value paris, feature_dim)
    # shape of value: (batch_size, no. of key-value pairs, feature_dim)
    def forward(self,query,key,value):
        d = query.shape[-1]
        attention_score = torch.bmm(query,key.transpose(2,1))/np.sqrt(d)
        # shape of output: (batch_size, no. of query, feature_dim)
        return torch.bmm(self.softmax(attention_score),value)        


# In[18]:


def transpose_kqv(X,num_heads):
    X = X.view(X.shape[0], X.shape[1], num_heads, -1)
    X = X.transpose(1,2)
    X = X.reshape(-1,X.shape[2],X.shape[3])
    return X


# In[26]:


class MultiHeadAttention(nn.Module):

    # we require num_hidden = num_heads*feature_dim
    def __init__(self,num_hidden,num_heads,bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.attention = ScaledDotProduction()
        self.Wq = nn.LazyLinear(num_hidden,bias=bias)
        self.Wk = nn.LazyLinear(num_hidden,bias=bias)
        self.Wv = nn.LazyLinear(num_hidden,bias=bias)
        self.Wo = nn.LazyLinear(num_hidden,bias=bias)
    
    def forward(self,query,key,value):
        # shape of Q/K/V: (batch_size*num_heads, no. of Q/K/V, num_hidden/num_heads)
        query = transpose_kqv(self.Wq(query),num_heads=self.num_heads)
        key = transpose_kqv(self.Wk(key),num_heads=self.num_heads)
        value = transpose_kqv(self.Wv(value),num_heads=self.num_heads)

        # shape of output: (batch_size*num_heads, no. of Q, num_hidden/num_heads)
        output = self.attention(query,key,value)
        
        output = output.reshape(-1,output.shape[1],self.num_hidden)
        return self.Wo(output)


# In[20]:


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=96, patch_size=16, num_hiddens=512):
        super().__init__()
        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x
        img_size, patch_size = _make_tuple(img_size), _make_tuple(patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1])
        self.conv = nn.LazyConv2d(num_hiddens, kernel_size=patch_size,
                                  stride=patch_size)

    def forward(self, X):
        # Output shape: (batch size, no. of patches, no. of channels)
        return self.conv(X).flatten(2).transpose(1, 2)


# In[28]:


class ViTMLP(nn.Module):
    def __init__(self,mlp_num_hidden,mlp_num_output,dropout=0.5):
        super().__init__()
        self.linear1 = nn.LazyLinear(mlp_num_hidden)
        self.relu = nn.ReLU()
        self.drpoout1 = nn.Dropout(dropout)
        self.linear2 = nn.LazyLinear(mlp_num_output)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,x):
        for module in self._modules.values():
            x = module(x)
        return x


# In[22]:


class ViTBlock(nn.Module):
    def __init__(self, num_hidden, norm_shape, mlp_num_hiddens,
                 num_heads, dropout, bias=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(norm_shape)
        self.attention = MultiHeadAttention(num_hidden, num_heads,
                                                bias)
        self.ln2 = nn.LayerNorm(norm_shape)
        self.mlp = ViTMLP(mlp_num_hiddens, num_hidden, dropout)

    def forward(self, X):
        X = X + self.attention(*([self.ln1(X)] * 3))
        return X + self.mlp(self.ln2(X))


# In[42]:


class ViT(nn.Module):
    def __init__(self, img_size, patch_size, num_hidden, mlp_num_hidden,
                 num_heads, num_blks, emb_dropout, blk_dropout, lr=0.1,
                 bias=False, num_classes=10):
        super().__init__()
        self.patch_embedding = PatchEmbedding(
            img_size, patch_size, num_hiddens)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_hiddens))
        num_steps = self.patch_embedding.num_patches + 1  # Add the cls token
        # Positional embeddings are learnable
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_steps, num_hiddens))
        self.dropout = nn.Dropout(emb_dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module(f"{i}", ViTBlock(
                num_hiddens, num_hiddens, mlp_num_hiddens,
                num_heads, blk_dropout, bias))
        self.head = nn.Sequential(nn.LayerNorm(num_hiddens),
                                  nn.Linear(num_hiddens, num_classes))

    def forward(self, X):
        X = self.patch_embedding(X)
        X = torch.cat((self.cls_token.expand(X.shape[0], -1, -1), X), 1)
        X = self.dropout(X + self.pos_embedding)
        for blk in self.blks:
            X = blk(X)
        return self.head(X[:, 0])


# In[43]:

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

    model.to(device)
    optimizer = optim.SGD(model.parameters(),lr=1e-2)
    trainer = train_and_valid(model,train_dataloader,valid_dataloader,criterion=nn.CrossEntropyLoss(),optimizer=optimizer,
                            output_cache = './ViT', device = device,file_name='ViT_FashionMNIST', batch_tqdm=False,
                            max_epochs=8)







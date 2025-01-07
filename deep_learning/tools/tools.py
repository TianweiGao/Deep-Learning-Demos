import torch
import torch.nn as nn
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import pdb

def train_and_valid(model,train_dataloader,valid_dataloader,criterion,optimizer,
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
            batch_loss = criterion(y_pred,y.to(device))
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss.detach().cpu().numpy()
            
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

def simpleClassifierTest(model,test_dataloader,device):
    score = 0.
    def criterion(y_pred,y):
        y1 = torch.argmax(y_pred,dim=1)
        return torch.sum(y1==y).cpu().numpy()

    for batch, (X,y) in enumerate(tqdm(test_dataloader)):
        y_pred = model(X.to(device))
        batch_loss = criterion(y_pred,y.to(device))
        score += batch_loss/y_pred.shape[0]
    
    score/= len(test_dataloader)

    print(f'Accuracy score is {score}')

def gpu():
    return torch.device(f'cuda')
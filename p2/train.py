from dataset import Dog_Cat_Dataset,DataLoader
from model import Model
from utils import accuracy_

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim 

np.random.seed(987)
torch.manual_seed(987)
if torch.cuda.is_available():
    torch.cuda.manual_seed(987)

def train(train_loader, model, criterion, optimizer, device):
    model.train()
    
    _iter, losses, acc = 0,0,0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        predicts = model(imgs)
        loss = criterion(predicts, labels)
        
        losses += loss.item()
        acc += accuracy_(predicts, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _iter += 1
        print('\t loss:%.3f acc:%.2f'%(losses/_iter, acc/_iter), end='  \r')
    
    print('\t train loss:%.3f acc:%.2f, '%(losses/_iter, acc/_iter))

@torch.no_grad()
def valid(train_loader, model, criterion, device):
    model.eval()
    
    _iter, losses, acc = 0,0,0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        predicts = model(imgs)
        loss = criterion(predicts, labels)
        
        losses += loss.item()
        acc += accuracy_(predicts, labels)
                
        _iter += 1
        print('\t loss:%.3f acc:%.2f'%(losses/_iter, acc/_iter), end='  \r')

    print('\t valid loss:%.3f acc:%.2f'%(losses/_iter, acc/_iter))
    return acc/_iter

if __name__=='__main__':
    
    # Datasets
    train_set = Dog_Cat_Dataset('./Cat_Dog_data_small/train')
    valid_set = Dog_Cat_Dataset('./Cat_Dog_data_small/valid')        

    train_loader = DataLoader(train_set,batch_size=64,shuffle=True)
    valid_loader = DataLoader(valid_set,batch_size=128)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Model
    model = Model()
    model.to(device)
    print(model)
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.05) #TODO:
    
    best_acc = 0
    for epoch in range(10): #TODO:
        print(f'Epoch {epoch}')
        
        train(train_loader, model, criterion, optimizer, device)
        
        valid_acc = valid(train_loader, model, criterion, device)
        
        if valid_acc > best_acc:
            print('\t save weights')
            torch.save(model.state_dict(),'best_model.pth')
            best_acc = valid_acc

import pandas as pd
import torch
import numpy as np
from sklearn import preprocessing
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

df = pd.read_csv('DS_hw6_p1.csv')

# preprocessing
df_x = preprocessing.scale(df.iloc[:,:-2], axis = 0)
train_x = df_x[:280, 1:]
test_x = df_x[280:, 1:]
train_y = df.iloc[:280, -2:].to_numpy()

# model structure
class linearRegression(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(linearRegression, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(n_feature, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output))

    def forward(self, x):
        out = self.linear(x)
        return out

# training settings
inputDim = 4        # takes variable 'x' 
outputDim = 2       # takes variable 'y'
learningRate = 0.01 
epochs = 100

model = linearRegression(inputDim, 18, outputDim)
optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()

# train
loss_all = []
for t in range(100):
    x = Variable(torch.from_numpy(train_x).float())
    y = Variable(torch.from_numpy(train_y).float())
    model.train()
    optimizer.zero_grad()
    prediction = model(x)
    loss = loss_func(prediction, y)
    loss.backward()
    loss_all.append(loss.detach().numpy())
    print('epoch = {}, loss = {}'.format(t,loss.detach().numpy()))
    optimizer.step()

# predict
model.eval()
x = Variable(torch.from_numpy(test_x).float())
pred_test = model(x)
pred_test


#plt.plot(loss_all)
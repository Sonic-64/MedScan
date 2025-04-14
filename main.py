import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as opt
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torchvision.utils import  make_grid
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
class ConvolutionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,6,3,1)
        self.conv2 = nn.Conv2d(6,16,3,1)
        self.fc1 = nn.Linear(5*5*16,120)
        self.fc2 = nn.Linear(120,64)
        self.fc3 = nn.Linear(64,120 )
    def forward(self,X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X,2,2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X,2,2)
        X = X.view(-1,16*5*5)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X,dim=1)

print(torch.__version__)

transform = transforms.ToTensor()
train_data = datasets.MNIST("../Data",train=True,download=True,transform=transform)
test_data = datasets.MNIST("../Data",train=False,download=True,transform=transform)

image,label = train_data[0]

torch.manual_seed(101)

train_loader = DataLoader(train_data,batch_size = 32,shuffle=True)
test_loader = DataLoader(test_data,batch_size = 128 , shuffle=False)
model = MultilayerPerceptron()
Conv1 = nn.Conv2d(1,6,3,1)
Conv2 = nn.Conv2d(6,16,3,1)
for i,(x_train,y_train) in enumerate(train_data):
    break

x=x_train.view(1,1,28,28)
print(x.shape)
x =F.relu(Conv1(x))
x =F.max_pool2d(x,2,2)
x =F.relu(Conv2(x))
x =F.max_pool2d(x,2,2)
print(x.shape)
x.view(-1,16*5*5)
print(type(x_train))













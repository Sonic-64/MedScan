import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import pydicom
import dicom2nifti
import nibabel as nib
import torch.optim as opt
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torchvision.utils import  make_grid
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Classifier(nn.Module):
    def __init__(self):
        super.__init__()
        self.conv1 = nn.Conv2d(1,8,3,1)
        self.conv2 = nn.Conv2d(8,16,3,1)
        self.fc1 = nn.Linear(5*5*16,120)
        self.fc2 = nn.Linear(120,90)
        self.fc3 = nn.Linear(90,10)
    def forward(self,X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X,2,2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d((X),2,2)
        X = X.view(-1,5*5*16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X,dim = 1)

train_loss = []
test_loss = []
train_corr = []
test_corr = []
print(torch.__version__)













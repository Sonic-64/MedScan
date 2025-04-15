import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import pydicom
import torch.optim as opt
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torchvision.utils import  make_grid
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


train_loss = []
test_loss = []
train_corr = []
test_corr = []
print(torch.__version__)













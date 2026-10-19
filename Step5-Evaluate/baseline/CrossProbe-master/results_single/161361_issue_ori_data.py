# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch 
import pandas as pd
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import statsmodels.api as sm
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

df = sm.datasets.get_rdataset("AirPassengers").data
df = df.shift(1)
df = df.dropna().reset_index(drop=True)
x = torch.from_numpy(df.index.values).reshape(-1,1).float().to(device)
data = torch.from_numpy(df['value'].values).float().to(device)
y = data.float().unsqueeze(1).to(device)  # shape (144, 1)
ones = torch.ones(data.size(0),1,requires_grad=False).to(device)# shape (144, 1)
w = torch.rand(2,1,requires_grad=True).to(device)
data_aug = torch.cat([x, ones], dim=1).to(device) # shape (144, 2)


for epoch in range(100):
    pred = data_aug @ w
    loss = torch.abs(y - pred).mean() 

    if w.grad is not None:
        w.grad.zero_()
    loss.backward()

    with torch.no_grad():
        w.data -= 0.01 * w.grad  
        w.grad.zero_()

    print(f"Epoch {epoch}: loss = {loss.item():.4f}")
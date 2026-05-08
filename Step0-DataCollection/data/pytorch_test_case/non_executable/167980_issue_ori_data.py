import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # hide GPU for this test

import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

device = "cpu"
print("Device:", device)

train_data = datasets.MNIST("data", train=True, download=True, transform=ToTensor())
test_data  = datasets.MNIST("data", train=False, download=True, transform=ToTensor())

train_loader = DataLoader(train_data, batch_size=80, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=80, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,10)
        )
    def forward(self,x):
        return self.layers(x)

model = Net().to(device)
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.SGD(model.parameters(), lr=0.01)

print("=== TRAIN ===")
for X,y in train_loader:
    X,y = X.to(device), y.to(device)
    pred = model(X)
    loss = loss_fn(pred,y)
    loss.backward()
    opt.step()
    opt.zero_grad()
    break

print("=== EVAL (CRASH HERE) ===")
with torch.no_grad():
    for batch, (X,y) in enumerate(test_loader):
        print("batch", batch)
        X,y = X.to(device), y.to(device)
        pred = model(X)       # → Kernel restart happens here
        break


### Versions

PyTorch version: 2.10.0.dev20251114+cu128
Is debug build: False
CUDA used to build PyTorch: 12.8
OS: Windows 11 Pro (10.0.26200)
Python: 3.12.12 (conda-forge)
GPU: NVIDIA GeForce RTX 5080 (sm_120)
NVIDIA driver: 581.42
torchvision: 0.25.0.dev20251116+cu128
numpy: 2.3.4
MKL: 2025.3.0 (conda-forge)

🔎 Notes

The crash occurs on CPU and CUDA

The crash also occurs in a clean conda environment

Shapes and model are correct, so this is likely a native backend failure

RTX 5080 + CUDA 12.8 + Nightly torch wheels may trigger new ATen failures
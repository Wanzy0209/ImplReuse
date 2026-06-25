import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = "cpu"
train_data = datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST("data", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_data, batch_size=80, shuffle=True)
test_loader = DataLoader(test_data, batch_size=80, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Flatten(), nn.Linear(784,256), nn.ReLU(), nn.Linear(256,256), nn.ReLU(), nn.Linear(256,10))
    def forward(self,x):
        return self.layers(x)

model = Net().to(device)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

for X,y in train_loader:
    X,y = X.to(device), y.to(device)
    pred = model(X)
    loss = loss_fn(pred,y)
    loss.backward()
    opt.step()
    opt.zero_grad()
    break

with torch.no_grad():
    for batch, (X,y) in enumerate(test_loader):
        X,y = X.to(device), y.to(device)
        pred = model(X)
        break
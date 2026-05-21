import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # hide GPU for this test

import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

device = "cpu"
print("Device:", device)

# Setup Data
train_data = datasets.MNIST("data", train=True, download=True, transform=ToTensor())
test_data  = datasets.MNIST("data", train=False, download=True, transform=ToTensor())

train_loader = DataLoader(train_data, batch_size=80, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=80, shuffle=True)

# Define Model leveraging the similar API pattern (ExtractImagePatches -> Unfold)
# The similar API tf.raw_ops.ExtractImagePatches extracts patches from images.
# In PyTorch, this is equivalent to nn.Unfold.
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Leverage nn.Unfold which corresponds to tf.raw_ops.ExtractImagePatches
        # kernel_size, dilation, padding, stride map to ksizes, rates, padding, strides
        self.unfold = nn.Unfold(kernel_size=3, dilation=1, padding=1, stride=1)
        
        # Calculate flattened size after unfolding
        # Input: (N, 1, 28, 28)
        # Unfold output channels: 1 * 3 * 3 = 9
        # Unfold output spatial dims: 28 * 28 = 784
        # Total features: 9 * 784 = 7056
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(7056, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Apply the operation similar to ExtractImagePatches
        x = self.unfold(x) 
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = Net().to(device)
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.SGD(model.parameters(), lr=0.01)

print("=== TRAIN ===")
for X, y in train_loader:
    X, y = X.to(device), y.to(device)
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    opt.step()
    opt.zero_grad()
    break

print("=== EVAL (CRASH HERE) ===")
with torch.no_grad():
    for batch, (X, y) in enumerate(test_loader):
        print("batch", batch)
        X, y = X.to(device), y.to(device)
        # The crash in the original issue happens during model inference
        pred = model(X)       
        break

print("Test completed successfully if no crash occurred.")
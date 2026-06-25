import torch
x = torch.arange(12)
y = x[::2]
z = y.view(3, 2)
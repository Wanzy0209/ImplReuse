import torch
from torch import nn

device = torch.device('mps')

with torch.amp.autocast(device_type=device.type):
    m = nn.ConvTranspose3d(16, 33, 3, stride=2)
    m.to(device)
    x = torch.randn(20, 16, 10, 50, 100).to(device)
    u = m(x)
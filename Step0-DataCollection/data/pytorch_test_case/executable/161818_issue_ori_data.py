import torch
from torch import nn

module = nn.Linear(8, 12)

padded = torch.rand(9, 8)
lengths = torch.as_tensor([5, 4])

with torch.autograd.set_detect_anomaly(True):
    out = module(padded)
    nopad = torch.nested.narrow(out, dim=1, start=0, length=lengths, layout=torch.jagged).contiguous().values()
    nopad.sum().backward()
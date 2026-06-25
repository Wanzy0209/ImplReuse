import torch
x = torch.randn(2, 1, 3)
dlpack = torch.to_dlpack(x)
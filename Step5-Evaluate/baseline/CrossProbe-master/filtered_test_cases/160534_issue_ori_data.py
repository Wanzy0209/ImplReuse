import torch
x = torch.randn(2, 2, requires_grad=True)
torch.utils.checkpoint.checkpoint(lambda y: y * 2, x)
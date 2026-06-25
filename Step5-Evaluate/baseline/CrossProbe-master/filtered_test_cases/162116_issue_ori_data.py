import torch
x = torch.empty((0,))
y = torch.empty(2,2)
try:
    torch.broadcast_tensors(x, y)
except RuntimeError as e:
    print(e)
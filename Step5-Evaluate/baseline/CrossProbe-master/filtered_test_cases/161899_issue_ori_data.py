import torch
x = torch.tensor([[1, 2], [3, 4]])
y = torch.full_like(x, 5)
print(y)
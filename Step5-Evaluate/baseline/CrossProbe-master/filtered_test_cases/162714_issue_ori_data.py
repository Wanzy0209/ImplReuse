import torch
x = torch.arange(8)
result = torch.tensor_split(x, (1, 6))
print(result)
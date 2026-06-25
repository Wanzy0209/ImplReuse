import torch
x = torch.randn(3, 4)
print(torch.is_storage(x))  # False
storage = x.storage()
print(torch.is_storage(storage))  # True
import torch
result = torch.aminmax(torch.tensor([1, -3, 5]))
print(result)
# This will fail:
# torch.return_types.aminmax(min=torch.tensor(-3), max=torch.tensor(5))
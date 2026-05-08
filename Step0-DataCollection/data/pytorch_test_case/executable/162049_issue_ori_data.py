import torch

x = torch.nested.nested_tensor(
    [torch.arange(0, n) for n in (10, 20, 30)],
    layout=torch.jagged,
)
print(x.max(dim=1).values)
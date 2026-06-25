import torch
x = torch.ones(10).requires_grad_(True)
y = torch.ones(10).requires_grad_(True)
z = (x**2).sum()
g = torch.autograd.grad(z, (x, y), allow_unused=True, materialize_grads=True, retain_graph=False, create_graph=False)
print(g[0].requires_grad)  # False
print(g[1].requires_grad)  # True (unexpected)
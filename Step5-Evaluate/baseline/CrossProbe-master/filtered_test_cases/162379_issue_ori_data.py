import torch
x = torch.tensor(2.0, requires_grad=True)
y = x**2
grad = torch.autograd.grad(y, x)
print(grad)
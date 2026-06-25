import torch
x = torch.tensor(0.0, requires_grad=True)
y = torch.tensor(0.0, requires_grad=True)
z = torch.atan2(y, x)
z.backward()
print(x.grad, y.grad)
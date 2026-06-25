import torch
x = torch.tensor([[1., 2., -torch.inf], [2., 1., -torch.inf]])
t = torch.tensor(1., dtype=torch.float32, requires_grad=True)
y = torch.logsumexp(x / t, dim=0)
z = y[y.isfinite()].mean()
z.backward()
print(f"t.grad={t.grad}")
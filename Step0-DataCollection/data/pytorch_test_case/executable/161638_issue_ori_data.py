import torch


x = torch.tensor([[1., 2., -torch.inf], [2., 1., -torch.inf]])
t = torch.tensor(1., dtype=torch.float32, requires_grad=True)
y = torch.logsumexp(x / t, dim=0)
print(f"{y=}")

z = y[y.isfinite()].mean() # All non-finite values excluded in the calculation of z
z.backward()
print(f"{t.grad=}")

t.grad = None
y2 = torch.logsumexp(x[:, :2] / t, dim=0) # If non-finite values are excluded before logsumexp, all works fine
print(f"{y2=}")
z2 = y2[y2.isfinite()].mean()
z2.backward()
print(f"{t.grad=}")
import warnings
warnings.simplefilter('error')
import torch
print(torch.__version__)
a, b = torch.rand((2, 32, 32))
a.requires_grad_()
optimizer = torch.optim.LBFGS([a])
loss_fn = lambda x, y: (x-y).pow(2).mean()

def closure():
    optimizer.zero_grad()
    loss = loss_fn(a, b)
    loss.backward()
    return loss

for i in range(100):
    optimizer.step(closure)
    print(i, loss_fn(a, b))
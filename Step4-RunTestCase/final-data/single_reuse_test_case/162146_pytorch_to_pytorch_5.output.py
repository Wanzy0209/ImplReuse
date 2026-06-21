import torch
torch.manual_seed(2025)


def foo(x):
    x[0].sin_()
    x[1].sin_()
    y = torch.zeros_like(x)
    y[2] = x[0]
    y[3] = x[1]
    # Adapted to use torch.any to verify the similar API
    return torch.any(y > 0)


# Handle environments where torch.compile is not available (PyTorch < 2.0)
if hasattr(torch, 'compile'):
    cfoo = torch.compile(foo)
else:
    cfoo = foo

x = torch.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], dtype=torch.float32)
cx = torch.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], dtype=torch.float32)
res = foo(x)
cres = cfoo(cx)
torch.testing.assert_close(res, cres)
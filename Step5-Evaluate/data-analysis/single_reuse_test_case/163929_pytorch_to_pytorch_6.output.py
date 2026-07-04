import torch

# Handle environments where torch.compile is not available (PyTorch < 2.0)
if not hasattr(torch, 'compile'):
    # Mock torch.compile to return the function unchanged
    torch.compile = lambda f: f

def foo(x, y):
    x.tan_()
    x = x.t()
    return torch.fmax(x, y)

torch.manual_seed(0)
x1 = torch.randn(4, 6)
x2 = x1.clone()
# y must match the shape of x after transpose (6, 4)
y = torch.randn(6, 4)

out1 = foo(x1, y)
cf = torch.compile(foo)
out2 = cf(x2, y)

torch.testing.assert_close(out1, out2)
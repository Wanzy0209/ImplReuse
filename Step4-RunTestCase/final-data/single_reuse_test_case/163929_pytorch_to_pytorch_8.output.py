import torch

# Handle environments where torch.compile is not available (PyTorch < 2.0)
if not hasattr(torch, 'compile'):
    torch.compile = lambda func, *args, **kwargs: func

def foo(x):
    x.tan_()
    x = x.t()
    return x.is_complex()

torch.manual_seed(0)
x1 = torch.randn(4, 6)
x2 = x1.clone()
out1 = foo(x1)
cf = torch.compile(foo)
out2 = cf(x2)
torch.testing.assert_close(out1, out2)
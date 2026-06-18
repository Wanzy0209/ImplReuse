import torch

def foo(x):
    x.tan_()
    # Adapt the test to verify the similar API torch.t
    x = x.t()
    return x

torch.manual_seed(0)
x1 = torch.randn(4, 6)
x2 = x1.clone()
out1 = foo(x1)
cf = torch.compile(foo)
out2 = cf(x2)
torch.testing.assert_close(out1, out2)
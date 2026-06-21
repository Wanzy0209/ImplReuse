import torch

def foo(x):
    x.tan_()
    x = x.t()
    return x.imag  # Fixed: imag is a property, not a method

torch.manual_seed(0)
# Using complex dtype to ensure torch.imag returns meaningful data
x1 = torch.randn(4, 6, dtype=torch.complex64)
x2 = x1.clone()

out1 = foo(x1)
cf = torch.compile(foo)
out2 = cf(x2)

torch.testing.assert_close(out1, out2)
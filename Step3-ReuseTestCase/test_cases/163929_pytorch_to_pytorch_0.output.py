import torch

def foo(x):
    x.tan_()
    x = x.t()
    return x.argmax()

torch.manual_seed(0)
x1 = torch.randn(4, 6)
x2 = x1.clone()

# Eager execution
out1 = foo(x1)

# Compiled execution (inductor backend)
cf = torch.compile(foo)
out2 = cf(x2)

# Verify results match
torch.testing.assert_close(out1, out2)
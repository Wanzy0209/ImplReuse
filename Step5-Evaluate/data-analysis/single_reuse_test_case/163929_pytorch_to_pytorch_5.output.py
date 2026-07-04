import torch

# Handle missing torch.compile for PyTorch versions < 2.0
if not hasattr(torch, 'compile'):
    # Define a mock compile function that acts as a pass-through
    def compile(func):
        return func
else:
    compile = torch.compile

def foo(x):
    x.tan_()
    x = x.t()
    return x.trace()

torch.manual_seed(0)
x1 = torch.randn(4, 6)
x2 = x1.clone()

out1 = foo(x1)
cf = compile(foo)
out2 = cf(x2)

torch.testing.assert_close(out1, out2)
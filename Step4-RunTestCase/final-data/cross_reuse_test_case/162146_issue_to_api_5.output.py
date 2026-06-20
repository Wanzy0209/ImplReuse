import torch

# Set seed for reproducibility
torch.manual_seed(2025)

# Fix: Handle environments where torch.compile is not available (PyTorch < 2.0)
if not hasattr(torch, 'compile'):
    torch.compile = lambda func: func

# Similar API pattern: A wrapper for torch.rand found in the codebase
def rand(*shape):
    return torch.rand(*shape).mul(16).add(1)

# Function containing the bug logic (in-place ops on slices + index_put)
def foo(x):
    x[0].sin_()
    x[1].sin_()
    y = torch.zeros_like(x)
    y[2] = x[0]
    y[3] = x[1]
    return y

# Compile the function
cfoo = torch.compile(foo)

# Generate inputs using the similar API pattern
x = rand(4, 3)
cx = x.clone()

# Execute both eager and compiled versions
res = foo(x)
cres = cfoo(cx)

# Assert that results match (this will fail if the bug is present)
torch.testing.assert_close(res, cres)
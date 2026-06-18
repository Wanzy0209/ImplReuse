import sys
import torch

def fn(x, n):
    if n == 0:
        return x
    # Adaptation: Use torch.all inside the recursive function to verify
    # its behavior within the compiled graph and respect for recursion limits.
    if torch.all(x > 0):
        return fn(x, n - 1) + 1
    return x

@torch.compile(backend="eager")
def outer(x):
    return fn(x, 1000)

sys.setrecursionlimit(10000000)
result = outer(torch.ones(3))

# Verify the result is as expected (1000 increments of 1.0)
assert torch.all(result == 1001)
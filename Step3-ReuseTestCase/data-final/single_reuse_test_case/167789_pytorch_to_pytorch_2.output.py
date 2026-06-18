import sys
import torch

def fn(x, n):
    if n == 0:
        return x
    # Adapted to use torch.prod (the similar API) inside the recursive logic
    return fn(x, n - 1) + torch.prod(x)

@torch.compile(backend="eager")
def outer(x):
    return fn(x, 1000)

# Set recursion limit as described in the bug report
sys.setrecursionlimit(10000000)

# Execute the test
input_tensor = torch.ones(3)
result = outer(input_tensor)

# Verify the result to ensure the API executed correctly
# prod(ones(3)) is 1.0, so result should be ones(3) + 1000.0
expected = input_tensor + 1000.0
assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"
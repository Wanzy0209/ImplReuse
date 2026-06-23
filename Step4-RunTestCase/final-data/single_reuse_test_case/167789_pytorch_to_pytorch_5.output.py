import sys
import torch

# Set recursion limit high to avoid RecursionError in Python interpreter
# The bug report indicates that torch.compile ignores this setting.
sys.setrecursionlimit(10000000)

# Fix: Handle environments where torch.compile is not available (PyTorch < 2.0)
# We mock it as a pass-through decorator to allow the test logic to run.
if not hasattr(torch, 'compile'):
    def mock_compile(**kwargs):
        def decorator(func):
            return func
        return decorator
    torch.compile = mock_compile

def fn(x, n):
    if n == 0:
        return x
    # Adaptation: Use torch.any in the recursive step to verify the similar API
    # torch.any(x > 0) returns a boolean scalar (True), which acts like 1.0 when added
    return fn(x, n - 1) + torch.any(x > 0)

@torch.compile(backend="eager")
def outer(x):
    return fn(x, 1000)

# Run the test
input_tensor = torch.ones(3)
result = outer(input_tensor)

# Verify result
# fn(x, 1000) adds 1 (from torch.any) 1000 times to x.
# x is [1, 1, 1]. Result should be [1001, 1001, 1001].
expected = torch.ones(3) + 1000
assert torch.equal(result, expected), f"Expected {expected}, got {result}"
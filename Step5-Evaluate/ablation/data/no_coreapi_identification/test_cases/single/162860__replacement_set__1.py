import torch

def inner(x):
    return x + 1

@torch.compile(backend="eager")
def fn(x):
    x = inner(x)
    return inner(x)

# Test case with torch.zeros replacing torch.ones
input_tensor = torch.zeros(3)
result = fn(input_tensor)

# Verify the result
# inner(x) adds 1. fn calls inner twice. So result = x + 2.
# Input is 0s. Result should be 2s.
expected = torch.full((3,), 2.0)
assert torch.equal(result, expected), f"Expected {expected}, got {result}"
import torch

def inner(x):
    return x + 1

@torch.compile(backend="eager")
def fn(x):
    x = inner(x)
    return inner(x)

# Replacement: torch.ones -> torch.matrix_power
# Original: fn(torch.ones(3))
# torch.matrix_power requires a square matrix. We use torch.eye to create a valid input matrix.
input_tensor = torch.matrix_power(torch.eye(3), 2)

result = fn(input_tensor)

# Assertions to verify the logic is preserved
# inner(x) adds 1, fn(x) calls inner twice, so result should be x + 2
# input_tensor is the identity matrix (torch.eye(3))
expected = torch.eye(3) + 2
assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"
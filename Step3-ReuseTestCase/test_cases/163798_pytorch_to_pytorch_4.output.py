import torch

# Test case for torch.all inside torch.compile
# Adapted from the tolist() bug report to verify similar behavior
@torch.compile(fullgraph=False, backend="eager")
def func(a):
    # torch.all returns a boolean tensor (0-dim)
    # We use it in a computation similar to the tolist() example
    u0 = torch.all(a)
    return a * u0

# Run the test
input_tensor = torch.tensor([1, 2])
result = func(input_tensor)
expected = input_tensor * torch.all(input_tensor)

# Assert correctness
assert torch.equal(result, expected)
print("Test passed.")
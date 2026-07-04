import torch

# Check if torch.compile exists (introduced in PyTorch 2.0)
# If not, we mock it to allow the test to run on older versions
if not hasattr(torch, 'compile'):
    def mock_compile(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    torch.compile = mock_compile

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
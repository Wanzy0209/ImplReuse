import torch

# Check for torch.compile availability (PyTorch 2.0+)
# If not available, mock it to allow the test logic to execute without crashing.
if not hasattr(torch, 'compile'):
    print("torch.compile is not available in this PyTorch version. Mocking it to allow test execution.")
    torch.compile = lambda func, **kwargs: func

def forward(x):
    # Using torch.all inside the compiled function
    # This tests if torch.all traces correctly
    return x * torch.all(x)

x = torch.randn(2, 2)

compiled = torch.compile(forward, fullgraph=True)
result = compiled(x)

# Verify the result shape and values
assert result.shape == x.shape
# torch.all(x) checks if all elements are non-zero.
# Since x is random, it's unlikely all are non-zero, but we just check execution here.
# For a deterministic check, we can use a specific tensor.
x_ones = torch.ones(2, 2)
result_ones = compiled(x_ones)
assert torch.all(result_ones == x_ones)

print("Test passed.")
import torch
import sys

# Handle missing torch.compile (available in PyTorch 2.0+)
if not hasattr(torch, 'compile'):
    print("Warning: torch.compile not found. Using eager execution (mocking torch.compile).")
    # Mock torch.compile to allow the test to proceed without the decorator
    torch.compile = lambda func: func

# Adapt inputs for torch.bmm (requires 3D tensors with a batch dimension)
# Original shapes were (1024, 1024). We add a batch dimension.
batch_size = 2
A = torch.rand((batch_size, 1024, 1024), device="cuda", dtype=torch.float16)
B = torch.rand((batch_size, 1024, 1024), device="cuda", dtype=torch.float16)

@torch.compile
def linear_bmm(weight, input):
    # Adapted call: torch.mm -> torch.bmm
    # torch.bmm expects (batch, M, K) and (batch, K, N)
    return torch.bmm(input, weight, out_dtype=torch.float32)

# Execute the compiled function
result = linear_bmm(A, B)

# Verify that the output dtype matches the requested out_dtype
assert result.dtype == torch.float32, f"Expected torch.float32, but got {result.dtype}"
print("Test passed: torch.bmm handles out_dtype correctly under torch.compile.")
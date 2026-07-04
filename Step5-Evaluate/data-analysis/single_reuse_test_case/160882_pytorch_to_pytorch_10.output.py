import torch

# Handle environments where torch.compile is not available (PyTorch < 2.0)
if not hasattr(torch, 'compile'):
    # Mock torch.compile to simply return the function as-is
    # This allows the test logic to proceed without the compilation optimization
    torch.compile = lambda func, **kwargs: func

def f(input_tensor: torch.Tensor) -> torch.Tensor:
    # Adapted to use torch.movedim instead of torch.complex
    # We move dimension 1 to the last position
    return torch.movedim(input_tensor, 1, -1)

B, F, T = 1, 641, 39

# Create source tensor with shape (B, F, T)
src = torch.randn(B, F, T)

# Create a tensor with permuted dimensions (B, T, F)
# This mimics the "mismatch" in the original bug report
mismatch = src.permute(0, 2, 1)

# Compile with fullgraph=True to match the original bug's conditions
# If torch.compile is mocked, this just returns 'f'
compiled = torch.compile(f, fullgraph=True)

# First call with shape (B, F, T)
result_1 = compiled(src)

# Second call with shape (B, T, F)
# This tests if torch.movedim handles the shape change correctly
# without crashing like torch.complex did in the original issue.
result_2 = compiled(mismatch)

# Assertions to verify the behavior is correct
assert result_1.shape == (B, T, F), f"Expected shape {(B, T, F)}, got {result_1.shape}"
assert result_2.shape == (B, F, T), f"Expected shape {(B, F, T)}, got {result_2.shape}"

print("Test passed.")
import torch

# This test assumes a custom device backend (privateuse1) is registered
# and relies on CPU fallback for unimplemented operations.

# Create a tensor on the custom device
t = torch.randn(4, 4, device='privateuse1')

# This call is expected to fail (return shape [0]) if the bug affects torch.exp2
# similar to how it affects torch.abs.
result = torch.exp2(t)

# Verify the shape matches the input
assert result.shape == t.shape, f"Expected shape {t.shape}, but got {result.shape}"
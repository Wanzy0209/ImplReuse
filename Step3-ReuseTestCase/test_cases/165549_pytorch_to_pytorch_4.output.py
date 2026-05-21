import torch

# Assuming a custom 'privateuse1' backend is registered and relies on CPU fallback
# for unimplemented operations, as described in the bug report.

# Create a tensor on the custom device
t = torch.randn(4, 4, device='privateuse1')

# Test torch.tan (similar to torch.abs)
# Bug: returns tensor with shape [0] instead of [4, 4]
result = torch.tan(t)

# Verify the shape is correct
assert result.shape == torch.Size([4, 4]), f"Expected shape [4, 4], but got {result.shape}"
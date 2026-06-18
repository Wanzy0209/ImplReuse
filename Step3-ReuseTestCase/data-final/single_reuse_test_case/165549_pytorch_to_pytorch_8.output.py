import torch

# This test assumes a custom backend is registered on 'privateuse1'
# and implements the minimal set of ops with CPU fallback.

# Create a tensor on the custom device
# Note: acos requires inputs in [-1, 1], so we clamp the random values
t = torch.randn(4, 4, device='privateuse1')
t = torch.clamp(t, -1.0, 1.0)

# This is the adapted test case for torch.acos
# Bug: returns tensor with shape [0] instead of [4, 4]
result = torch.acos(t)

# Verify the shape is correct
assert result.shape == torch.Size([4, 4]), f"Expected shape [4, 4], but got {result.shape}"
import torch
import sys

# Check if the 'privateuse1' backend is registered.
# If not, we cannot run the test as it relies on specific backend behavior.
try:
    torch.empty(1, device='privateuse1')
except RuntimeError as e:
    if "Invalid device string" in str(e):
        print("Skipping test: 'privateuse1' backend is not registered.")
        sys.exit(0)
    raise

# Create a tensor on the custom device
t = torch.randn(4, 4, device='privateuse1')

# Test torch.tan (similar to torch.abs)
# Bug: returns tensor with shape [0] instead of [4, 4]
result = torch.tan(t)

# Verify the shape is correct
assert result.shape == torch.Size([4, 4]), f"Expected shape [4, 4], but got {result.shape}"
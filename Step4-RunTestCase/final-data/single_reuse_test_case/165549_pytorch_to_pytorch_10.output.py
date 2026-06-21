import torch
import sys

# This test assumes a custom device backend (privateuse1) is registered
# and relies on CPU fallback for unimplemented operations.

try:
    # Create a tensor on the custom device
    t = torch.randn(4, 4, device='privateuse1')
except RuntimeError as e:
    # Handle the case where the custom backend is not registered in the environment
    if "Invalid device string" in str(e):
        print("Skipping test: 'privateuse1' backend is not registered in this environment.")
        sys.exit(0)
    else:
        raise

# This is the failing case adapted for torch.log10
# Bug: returns tensor with shape [0] instead of [4, 4]
result = torch.log10(t)

# Assertion to verify the fix
assert result.shape == torch.Size([4, 4]), f"log10 returned incorrect shape: {result.shape}"

# The following cases are noted to work correctly in the bug report:
# t.log10_()  # In-place works
# torch.log10(t, out=pre_allocated_tensor) # Out variant works
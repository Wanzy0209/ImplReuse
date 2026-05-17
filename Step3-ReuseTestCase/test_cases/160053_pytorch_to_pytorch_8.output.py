import torch
import torch.special

# Test case adapted for torch.special.gammaincc
# The original bug report highlighted an issue with 4D inputs in F.pad.
# We verify that the similar API, torch.special.gammaincc, handles 4D inputs correctly.

# Create 4D inputs with positive values (required for gammaincc domain)
a = torch.rand(2, 2, 2, 2)
b = torch.rand(2, 2, 2, 2)

# Execute the function
result = torch.special.gammaincc(a, b)

# Verify the output is valid and shape is preserved
assert result is not None
assert result.shape == a.shape, f"Expected shape {a.shape}, but got {result.shape}"
print("Test passed: torch.special.gammaincc works with 4D inputs.")
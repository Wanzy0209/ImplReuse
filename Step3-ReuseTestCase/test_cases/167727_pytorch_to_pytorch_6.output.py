import torch
import torch.nn.functional as F

# Test case adapted for torch.nn.functional.hardshrink
# Using the large tensor dimensions from the original bug report to check for similar issues

# Create a large complex tensor on MPS (similar to the failing case in addmm)
input_tensor = torch.rand((64, 10000), dtype=torch.complex64, device="mps")

# Apply hardshrink on MPS
out_mps = F.hardshrink(input_tensor, lambd=0.5)

# Apply hardshrink on CPU for reference
out_cpu = F.hardshrink(input_tensor.cpu(), lambd=0.5)

# Verify results match
torch.testing.assert_close(out_mps.cpu(), out_cpu)
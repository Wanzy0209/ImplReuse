import torch
import torch.nn.functional as F

# Test case for torch.nn.functional.softshrink with large complex64 tensors on MPS
# Adapted from the torch.addmm bug report (Issue ID: 167727)

# Create a large tensor similar to the failing case in the original bug
input_tensor = torch.rand((64, 10000), dtype=torch.complex64, device="mps")

# Apply softshrink on MPS
out_mps = F.softshrink(input_tensor, lambd=0.5)

# Apply softshrink on CPU for reference
out_cpu = F.softshrink(input_tensor.cpu(), lambd=0.5)

# Verify correctness
torch.testing.assert_close(out_mps.cpu(), out_cpu)
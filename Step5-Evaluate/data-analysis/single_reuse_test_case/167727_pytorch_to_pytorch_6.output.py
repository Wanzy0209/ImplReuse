import torch
import torch.nn.functional as F

# Test case adapted for torch.nn.functional.hardshrink
# Using the large tensor dimensions from the original bug report to check for similar issues

# Check for MPS availability
if torch.backends.mps.is_available():
    # Create a large complex tensor on CPU first, then move to MPS
    # This works around the 'aten::empty.memory_format' not being implemented
    # for complex64 directly on MPS
    input_tensor = torch.rand((64, 10000), dtype=torch.complex64)
    input_tensor_mps = input_tensor.to("mps")

    # Apply hardshrink on MPS
    out_mps = F.hardshrink(input_tensor_mps, lambd=0.5)

    # Apply hardshrink on CPU for reference
    out_cpu = F.hardshrink(input_tensor, lambd=0.5)

    # Verify results match
    torch.testing.assert_close(out_mps.cpu(), out_cpu)
else:
    print("MPS device not available, skipping test.")
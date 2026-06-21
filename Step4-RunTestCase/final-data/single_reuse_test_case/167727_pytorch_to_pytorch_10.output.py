import torch
import sys

# Test case for torch.fmax adapted from the addmm large tensor bug report.
# The original bug involved incorrect results on MPS for large complex64 tensors.
# We verify torch.fmax with similar large tensor dimensions and dtype.

try:
    # Setup large tensors similar to the failing case in the bug report
    # Shape (64, 10000) corresponds to the large dimension 'K' in the original addmm failure
    a = torch.rand((64, 10000), dtype=torch.complex64, device="mps")
    b = torch.rand((64, 10000), dtype=torch.complex64, device="mps")
except (NotImplementedError, RuntimeError, AttributeError) as e:
    # Handle cases where MPS is not available or does not support complex64
    print(f"Skipping test: MPS backend or complex64 support not available. Error: {e}")
    sys.exit(0)

# Compute on MPS
out_mps = torch.fmax(a, b)

# Compute on CPU for reference
out_cpu = torch.fmax(a.cpu(), b.cpu())

# Verify that MPS results match CPU results
torch.testing.assert_close(out_mps.cpu(), out_cpu)
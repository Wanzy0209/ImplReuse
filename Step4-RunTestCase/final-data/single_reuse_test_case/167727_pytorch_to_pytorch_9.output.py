import torch

# Check if MPS is available before running the test
# The error occurs because the MPS backend is not supported on this system (likely Linux).
if torch.backends.mps.is_available():
    # Test case for torch.fmin adapted from the torch.addmm bug report.
    # Note: torch.fmin requires real-valued dtypes, so dtype is adapted to float32.

    # success (small tensors)
    a = torch.rand((64, 300), dtype=torch.float32, device="mps")
    b = torch.rand((64, 300), dtype=torch.float32, device="mps")
    out = torch.fmin(a, b)
    torch.testing.assert_close(out.cpu(), torch.fmin(a.cpu(), b.cpu()))

    # fails (large tensors - adapted dimensions from original bug)
    # Using large dimensions similar to the failing addmm case to check for numerical issues
    a = torch.rand((64, 10000), dtype=torch.float32, device="mps")
    b = torch.rand((64, 10000), dtype=torch.float32, device="mps")
    out = torch.fmin(a, b)
    torch.testing.assert_close(out.cpu(), torch.fmin(a.cpu(), b.cpu()))
else:
    print("Skipping test: MPS backend is not available on this system.")
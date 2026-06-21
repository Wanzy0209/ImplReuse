import torch

# Check for MPS availability to ensure the test runs on the correct backend
if not torch.backends.mps.is_available():
    print("MPS is not available. Skipping test.")
else:
    # Create a large tensor (>4GB) similar to the bug report.
    # Shape (2, (1 << 31) + 5) ensures the second diagonal element (a[1, 1])
    # is located beyond the 4GB boundary where the fillBuffer bug occurs.
    a = torch.ones(2, (1 << 31) + 5, dtype=torch.int8, device='mps')

    # Calculate the trace. The trace sums the diagonal elements.
    # Expected: 1 + 1 = 2
    # Buggy behavior: 1 + 0 = 1 (because a[1,1] is not filled correctly due to the buffer size)
    result = torch.trace(a)

    print(f"Trace result: {result}")
    assert result == 2, f"Expected trace to be 2, but got {result}"
import torch
import torch.nn.functional as F

def f(xs):
    # Adapted to use the similar API: torch.nn.functional.avg_pool3d
    # Input xs is expected to be 5D (N, C, D, H, W) for avg_pool3d
    return F.avg_pool3d(xs, kernel_size=2)

def backend(gm, inps):
    gm.print_readable()
    return gm

# Check for CUDA availability to handle environment issues
if torch.cuda.is_available():
    # Fix: torch.device is not a context manager in this PyTorch version.
    # Use torch.cuda.device instead to set the current device context.
    with torch.cuda.device("cuda"):
        # Create a 5D tensor suitable for avg_pool3d
        xs = torch.randn(1, 1, 2, 2, 2, device="cuda")

        # Eager execution
        f(xs)

        # Compiled execution
        # This verifies if the similar API has the same issue under torch.device context
        torch.compile(f, backend=backend)(xs)
else:
    print("CUDA is not available. Skipping test.")
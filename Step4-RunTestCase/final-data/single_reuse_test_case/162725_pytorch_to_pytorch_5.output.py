import torch
import sys

# Define the function using the similar API (torch.einsum)
def fn(x, w):
    # Using einsum to perform a contraction on 5D tensors, 
    # mimicking the complexity of conv_transpose3d inputs.
    # x: (Batch, InChannels, Depth, Height, Width)
    # w: (InChannels, OutChannels, kD, kH, kW)
    # Result: (Batch, OutChannels, kD, kH, kW)
    return torch.einsum('ncdhw, cijk -> nij', x, w)

# Check if torch.compile is available (PyTorch 2.0+)
if not hasattr(torch, 'compile'):
    print("Skipping test: torch.compile is not available (requires PyTorch 2.0+)")
    sys.exit(0)

# Setup inputs (mimicking the sample inputs from the original bug report)
# Using CUDA and float32 as per the original report
device = "cuda"
dtype = torch.float32

# Check if CUDA is available
if not torch.cuda.is_available():
    print("Skipping test: CUDA is not available")
    sys.exit(0)

# Create sample tensors
# Batch=2, InChannels=4, D=5, H=5, W=5
x = torch.randn(2, 4, 5, 5, 5, device=device, dtype=dtype)
# InChannels=4, OutChannels=8, kD=3, kH=3, kW=3
w = torch.randn(4, 8, 3, 3, 3, device=device, dtype=dtype)

# Compile the function using the same backend and mode as the original bug
compiled = torch.compile(fn, backend="inductor", mode="max-autotune")

# Run eager and compiled versions
res1 = fn(x, w)
res2 = compiled(x, w)

# Assert results are close to check for consistency
torch.testing.assert_close(res1, res2)
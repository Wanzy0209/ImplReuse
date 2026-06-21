import torch
import sys

# Check for XPU availability to prevent RuntimeError
# This handles the environment issue where PyTorch is not linked with XPU support
if not (hasattr(torch, 'xpu') and torch.xpu.is_available()):
    print("Skipping test: XPU device is not available or PyTorch is not compiled with XPU support.")
    sys.exit(0)

def any_func(x):
    # Using the similar API: torch.any
    return torch.any(x)

# Setup data on XPU as per the original bug report
x = torch.randn(128).to("xpu")

# Test eager mode
out = any_func(x)
print("eager mode passed")

# Test compiled mode (where the segmentation fault occurred in the original issue)
any_func_compiled = torch.compile(any_func)
out = any_func_compiled(x)
print("torch.compile passed")
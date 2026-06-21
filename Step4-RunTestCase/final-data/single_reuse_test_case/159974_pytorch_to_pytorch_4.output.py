import torch

def all_func(x):
    # Using torch.all as the target API to test compilation
    return torch.all(x > 0)

# Check for XPU availability and fallback to CPU if necessary
device = "xpu"
try:
    # Check if xpu module exists and is available
    if not (hasattr(torch, 'xpu') and torch.xpu.is_available()):
        raise RuntimeError("XPU not available")
except (AttributeError, RuntimeError):
    print("Warning: XPU device not available. Falling back to CPU for testing.")
    device = "cpu"

# Setup data
x = torch.randn(128).to(device)

# Test eager mode
out_eager = all_func(x)
print("eager mode passed")

# Test compiled mode
all_func_compiled = torch.compile(all_func)
out_compiled = all_func_compiled(x)
print("torch.compile passed")

# Verify results match
assert torch.equal(out_eager, out_compiled), "Results mismatch between eager and compiled modes"
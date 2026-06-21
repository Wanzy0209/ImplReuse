import torch

def prod_func(x):
    # Using the similar API: torch.prod
    return torch.prod(x)

# Determine the available device to avoid RuntimeError
# Fallback to CUDA or CPU if XPU is not supported
if hasattr(torch, 'xpu') and torch.xpu.is_available():
    device = "xpu"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Running test on device: {device}")

# Setup data on the selected device
x = torch.randn(128).to(device)

# Test eager mode
out = prod_func(x)
print("eager mode passed")

# Test compiled mode
prod_func_compiled = torch.compile(prod_func)
out = prod_func_compiled(x)
print("torch.compile passed")

# Verify results match (if no segfault occurs)
assert torch.allclose(prod_func(x), out)
import torch

def func_nojit(x):
    # Adapted to use torch.amin instead of torch.full
    return torch.amin(x)

# Fix for AttributeError: module 'torch' has no attribute 'compile'
# This handles environments with PyTorch < 2.0 by falling back to torch.jit.script
if hasattr(torch, 'compile'):
    func_jit = torch.compile(func_nojit)
elif hasattr(torch, 'jit') and hasattr(torch.jit, 'script'):
    print("Warning: torch.compile not found (requires PyTorch 2.0+). Using torch.jit.script as fallback.")
    func_jit = torch.jit.script(func_nojit)
else:
    print("Warning: No compilation method found. Running in eager mode for both.")
    func_jit = func_nojit

# Test inputs with float64 dtype, mirroring the original bug report conditions
x1 = torch.tensor([5.0, 2.0], dtype=torch.float64)
x2 = torch.tensor([10.0, 1.0], dtype=torch.float64)

# Run eager mode
res_eager_1 = func_nojit(x1)
res_eager_2 = func_nojit(x2)

# Run compiled mode
res_compiled_1 = func_jit(x1)
res_compiled_2 = func_jit(x2)

# Verify results match
assert torch.equal(res_eager_1, res_compiled_1), f"Mismatch on first call: {res_eager_1} vs {res_compiled_1}"
assert torch.equal(res_eager_2, res_compiled_2), f"Mismatch on second call (potential caching bug): {res_eager_2} vs {res_compiled_2}"

print("All assertions passed.")
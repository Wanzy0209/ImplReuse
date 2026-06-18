import torch

def func_nojit(x):
    # Adapted from torch.full to torch.amax
    return torch.amax(x)

func_jit = torch.compile(func_nojit)

# Inputs with dtype=torch.float64 to match the bug report context
x1 = torch.tensor([1.0, 5.0], dtype=torch.float64)
x2 = torch.tensor([2.0, 10.0], dtype=torch.float64)

# Run non-compiled
print("No JIT:")
print(func_nojit(x1))
print(func_nojit(x2))

# Run compiled
print("JIT:")
print(func_jit(x1))
print(func_jit(x2))

# Assertions to verify correctness
assert torch.equal(func_nojit(x1), func_jit(x1)), "Mismatch on first call"
assert torch.equal(func_nojit(x2), func_jit(x2)), "Mismatch on second call"
# Ensure the compiled function didn't cache the result of the first call
assert not torch.equal(func_jit(x1), func_jit(x2)), "Compiled function cached the first result"
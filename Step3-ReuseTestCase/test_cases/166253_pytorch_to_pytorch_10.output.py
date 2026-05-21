import torch
import sys

def func_nojit(x):
    # Adapted from torch.full to torch.rand_like
    # Original: torch.full((2, ), x, dtype=torch.float64)
    # Adapted: torch.rand_like(x, dtype=torch.float64)
    # We use x as the reference tensor for shape and dtype.
    return torch.rand_like(x, dtype=torch.float64)

func_jit = torch.compile(func_nojit)

# Adapted inputs to be tensors of shape (2,) to match the original output shape
# and ensure the test covers non-scalar tensors.
x1 = torch.tensor([5.0, 5.0], dtype=torch.float64)
x2 = torch.tensor([10.0, 10.0], dtype=torch.float64)

print("Testing torch.rand_like with torch.compile and dtype=torch.float64")

for name, func in [("No JIT", func_nojit), ("JIT", func_jit)]:
    print(f"\n{name}:")
    out1 = func(x1)
    out2 = func(x2)
    print(out1)
    print(out2)

    # Assertions
    # 1. Check dtype
    assert out1.dtype == torch.float64, f"{name}: Expected float64, got {out1.dtype}"
    assert out2.dtype == torch.float64, f"{name}: Expected float64, got {out2.dtype}"

    # 2. Check for caching bug (outputs should be different for random ops)
    # Note: There is a tiny probability of collision, but for a test case, 
    # identical outputs indicate a caching/state bug.
    if name == "JIT":
        try:
            assert not torch.equal(out1, out2), \
                "JIT: torch.rand_like returned identical values (possible caching bug like torch.full)"
        except AssertionError as e:
            print(f"AssertionError: {e}")
            sys.exit(1)

print("\nTest passed successfully.")
import torch

def func_nojit(x):
    # Using the similar API: torch.special.spherical_bessel_j0
    return torch.special.spherical_bessel_j0(x)

func_jit = torch.compile(func_nojit)

# Test inputs matching the original bug report (float64)
x1 = torch.tensor(5.0, dtype=torch.float64)
x2 = torch.tensor(10.0, dtype=torch.float64)

# Run eager mode
res_eager_1 = func_nojit(x1)
res_eager_2 = func_nojit(x2)

# Run compiled mode
res_jit_1 = func_jit(x1)
res_jit_2 = func_jit(x2)

# Assertions to verify correctness and detect the specific caching bug
# 1. Compiled results should match eager results
assert torch.allclose(res_jit_1, res_eager_1), "Compiled result for x1 does not match eager execution"
assert torch.allclose(res_jit_2, res_eager_2), "Compiled result for x2 does not match eager execution"

# 2. Results for different inputs should be different (checking for the 'stuck value' bug)
# If the bug similar to torch.full exists, res_jit_2 would be equal to res_jit_1
assert not torch.allclose(res_jit_1, res_jit_2), "Compiled results for x1 and x2 are identical; potential caching bug detected"

print("Test passed.")
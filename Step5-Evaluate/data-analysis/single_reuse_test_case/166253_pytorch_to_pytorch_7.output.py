import torch

# Handle environments where torch.compile (PyTorch 2.0+) is not available
if not hasattr(torch, 'compile'):
    # Mock torch.compile to be a pass-through function
    # This allows the test to run without crashing, effectively testing only the eager mode
    torch.compile = lambda f, *args, **kwargs: f

def func_nojit(x):
    # Adapted to use torch.argmax instead of torch.full
    return torch.argmax(x)

func_jit = torch.compile(func_nojit)

# Test inputs with dtype=torch.float64, similar to the original bug report
# x1 has max value 5.0 at index 1
x1 = torch.tensor([1.0, 5.0, 3.0], dtype=torch.float64)
# x2 has max value 10.0 at index 0
x2 = torch.tensor([10.0, 2.0, 8.0], dtype=torch.float64)

# Expected results
expected_x1 = 1
expected_x2 = 0

# Test non-compiled function
res1_nojit = func_nojit(x1)
res2_nojit = func_nojit(x2)
assert res1_nojit == expected_x1, f"func_nojit failed for x1: expected {expected_x1}, got {res1_nojit}"
assert res2_nojit == expected_x2, f"func_nojit failed for x2: expected {expected_x2}, got {res2_nojit}"

# Test compiled function
res1_jit = func_jit(x1)
res2_jit = func_jit(x2)

# Verify that the compiled function returns the correct results for the specific inputs
# and does not cache the result from the first call (which was the bug in torch.full)
assert res1_jit == expected_x1, f"func_jit failed for x1: expected {expected_x1}, got {res1_jit}"
assert res2_jit == expected_x2, f"func_jit failed for x2: expected {expected_x2}, got {res2_jit}"

print("All tests passed.")
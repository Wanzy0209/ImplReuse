import torch

def func_nojit(x, y):
    # Adapted to use torch.fmin instead of torch.full
    return torch.fmin(x, y)

func_jit = torch.compile(func_nojit)

# Define inputs with dtype=torch.float64 to match the original bug context
x1 = torch.tensor(5.0, dtype=torch.float64)
y1 = torch.tensor(3.0, dtype=torch.float64)

x2 = torch.tensor(10.0, dtype=torch.float64)
y2 = torch.tensor(8.0, dtype=torch.float64)

# Expected results
expected_1 = torch.tensor(3.0, dtype=torch.float64)
expected_2 = torch.tensor(8.0, dtype=torch.float64)

# Test non-jit version
res_nojit_1 = func_nojit(x1, y1)
res_nojit_2 = func_nojit(x2, y2)

# Test jit version
res_jit_1 = func_jit(x1, y1)
res_jit_2 = func_jit(x2, y2)

# Verify results
# The original bug caused the second call to return the result of the first call.
# We assert here that torch.fmin updates correctly with new float64 inputs.
assert torch.equal(res_nojit_1, expected_1), "NoJIT: First call failed"
assert torch.equal(res_nojit_2, expected_2), "NoJIT: Second call failed"

assert torch.equal(res_jit_1, expected_1), "JIT: First call failed"
assert torch.equal(res_jit_2, expected_2), "JIT: Second call failed (potential caching bug)"

print("Test passed.")
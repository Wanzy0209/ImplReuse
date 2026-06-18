import torch
import torch.special

def func_nojit(x):
    # Using the similar API: torch.special.airy_ai
    return torch.special.airy_ai(x)

# Compile the function to check for caching issues similar to the torch.full bug
func_jit = torch.compile(func_nojit)

# Define inputs with dtype=torch.float64, which triggered the original bug
x1 = torch.tensor(5.0, dtype=torch.float64)
x2 = torch.tensor(10.0, dtype=torch.float64)

# Get expected results from eager execution
expected_1 = func_nojit(x1)
expected_2 = func_nojit(x2)

# Get results from compiled execution
result_1 = func_jit(x1)
result_2 = func_jit(x2)

# Verify that the compiled function produces correct results for both calls
# and specifically that the second call does not return the cached result of the first call.
assert torch.allclose(result_1, expected_1), "First compiled call result mismatch"
assert torch.allclose(result_2, expected_2), "Second compiled call result mismatch"

# Explicitly check against the bug pattern: result_2 should not equal result_1
assert not torch.allclose(result_2, result_1), "Bug detected: Second call returned result of first call"

print("Test passed.")
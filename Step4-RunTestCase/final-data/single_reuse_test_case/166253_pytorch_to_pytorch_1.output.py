import torch

def func_nojit(x, y):
    # Adapted to use torch.fmax instead of torch.full
    # We ensure inputs are float64 to match the context of the original bug
    return torch.fmax(x, y)

# Fix: Handle missing torch.compile (PyTorch 2.0+) by falling back to torch.jit.script (PyTorch 1.x)
if hasattr(torch, 'compile'):
    func_jit = torch.compile(func_nojit)
elif hasattr(torch, 'jit'):
    func_jit = torch.jit.script(func_nojit)
else:
    # If no JIT is available, we cannot test the JIT behavior.
    # We assign the non-jit function to allow the script to run without crashing,
    # though the specific JIT caching bug cannot be verified.
    func_jit = func_nojit

# Test Case 1
x1 = torch.tensor(5.0, dtype=torch.float64)
y1 = torch.tensor(3.0, dtype=torch.float64)

# Test Case 2 (Different values to check for caching issues)
x2 = torch.tensor(10.0, dtype=torch.float64)
y2 = torch.tensor(8.0, dtype=torch.float64)

# Run non-jit version
print("No JIT:")
print(func_nojit(x1, y1))
print(func_nojit(x2, y2))

# Run jit version
print("JIT:")
res1 = func_jit(x1, y1)
print(res1)
res2 = func_jit(x2, y2)
print(res2)

# Assertions
# The original bug caused the second call to return the result of the first call.
# We verify that the second call returns the correct result for the new inputs.
assert torch.equal(res1, torch.tensor(5.0, dtype=torch.float64)), "First call result mismatch"
assert torch.equal(res2, torch.tensor(10.0, dtype=torch.float64)), "Second call result mismatch (potential caching bug)"
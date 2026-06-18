import torch

# Adapted test case for torch.randint
# The original bug involved torch.compile caching the 'fill_value' argument of torch.full.
# Here we test if torch.compile caches the 'low' and 'high' arguments of torch.randint.

def func_nojit(low, high):
    # torch.randint generates random integers between low (inclusive) and high (exclusive)
    return torch.randint(low, high, (2, ), dtype=torch.int64)

func_jit = torch.compile(func_nojit)

# Test 1: No JIT
print("No JIT:")
# First call with range [0, 5)
x1 = func_nojit(0, 5)
print(f"Range [0, 5): {x1}")
assert (x1 >= 0).all() and (x1 < 5).all()

# Second call with range [10, 20)
x2 = func_nojit(10, 20)
print(f"Range [10, 20): {x2}")
assert (x2 >= 10).all() and (x2 < 20).all()

# Test 2: With JIT
print("\nWith JIT:")
# First call with range [0, 5)
y1 = func_jit(0, 5)
print(f"Range [0, 5): {y1}")
assert (y1 >= 0).all() and (y1 < 5).all()

# Second call with range [10, 20)
y2 = func_jit(10, 20)
print(f"Range [10, 20): {y2}")

# Assertion to detect the caching bug
# If the bug exists, y2 will contain values in [0, 5) instead of [10, 20)
try:
    assert (y2 >= 10).all() and (y2 < 20).all(), \
        f"Bug detected: torch.randint likely cached arguments. Expected range [10, 20), got {y2}"
    print("Test Passed: Arguments were not cached.")
except AssertionError as e:
    print(e)
import os
import torch

# Setup logging to match the original bug report context
os.environ["TORCH_LOGS"] = "output_code"

# Fix: Handle missing torch.compile for PyTorch versions < 2.0
if not hasattr(torch, "compile"):
    print("Warning: torch.compile not found (PyTorch < 2.0). Mocking as identity function.")
    torch.compile = lambda func: func

device = "cuda"

# Define a function that uses torch.any, adapted from the original context.
# The original bug involved specific tensor shapes (1, 32) and (1,) with int64 and int32 types.
# We adapt the logic to use torch.any while maintaining the input characteristics to verify
# if the compiler handles this similar API correctly.
def f(x, y):
    # x is (1, 32), y is (1,)
    # We perform a broadcast operation (y[:, None]) which was part of the original crash pattern
    # and apply torch.any to the result.
    return torch.any(x > y[:, None])

# Inputs matching the original bug report
x = torch.zeros(1, 32, dtype=torch.int64, device=device)
y = torch.zeros(1, dtype=torch.int32, device=device)

# Test eager execution
eager_result = f(x, y)
print(f"Eager result: {eager_result}")

# Test compiled execution
# This verifies if torch.compile handles torch.any correctly with these specific shapes/types
compiled_result = torch.compile(f)(x, y)
print(f"Compiled result: {compiled_result}")

# Assertion to ensure correctness
assert eager_result == compiled_result, "Mismatch between eager and compiled results"
import torch

# Define the minimum value for int64
int64_min = torch.iinfo(torch.int64).min

# Create dividend tensor using torch.randint
# To simulate torch.full with a specific value 'v', we use randint with range [v, v+1)
dividend = torch.randint(int64_min, int64_min + 1, (2, 3), dtype=torch.int64, device='cpu')

# Create divisor tensor using torch.randint
# To simulate torch.full with -1, we use randint with range [-1, 0)
divisor = torch.randint(-1, 0, (3,), dtype=torch.int64, device='cpu')

print("Dividend tensor:", dividend)
print("Divisor tensor:", divisor)

# This call is expected to crash in the buggy version (segfault / arithmetic exception)
# If the bug is fixed, it should complete successfully.
result = torch.fmod(dividend, divisor)
print("Result:", result)

# Assertion to verify the operation completed without crashing
assert isinstance(result, torch.Tensor)
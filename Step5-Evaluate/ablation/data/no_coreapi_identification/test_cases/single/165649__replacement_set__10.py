import torch

# Setup reference tensors for torch.rand_like
# torch.rand_like requires an input tensor to determine shape and dtype
ref_dividend = torch.empty((2, 3), dtype=torch.int64)
ref_divisor = torch.empty((3,), dtype=torch.int64)

# Replacement: torch.full -> torch.rand_like
dividend = torch.rand_like(ref_dividend)
divisor = torch.rand_like(ref_divisor)

# Logic: Set specific values to trigger the bug (INT64_MIN and -1)
# We must overwrite the random values to reproduce the specific crash condition
dividend.fill_(torch.iinfo(torch.int64).min)
divisor.fill_(-1)

print("Dividend tensor:", dividend)
print("Divisor tensor:", divisor)

# API Under Test: torch.fmod
# This call is expected to crash in the buggy version (Issue 165649)
result = torch.fmod(dividend, divisor)
print("Result:", result)

# Assertion to verify the operation completed successfully (no crash)
# and the result shape is as expected (broadcasting (2, 3) and (3,) -> (2, 3))
assert result.shape == (2, 3)
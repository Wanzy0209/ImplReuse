import torch

# Use torch.amax to construct the dividend tensor.
# Original logic: torch.full((2, 3), torch.iinfo(torch.int64).min, ...)
# Replacement logic: Create a (2, 3, 1) tensor containing INT64_MIN and reduce the last dimension.
min_val = torch.iinfo(torch.int64).min
# Create a scalar tensor and expand it to the required shape (2, 3, 1)
# We use torch.tensor and expand to avoid using torch.full.
dividend_input = torch.tensor(min_val, dtype=torch.int64, device='cpu').reshape(1, 1, 1).expand(2, 3, 1)
dividend = torch.amax(dividend_input, dim=-1)

# Use torch.amax to construct the divisor tensor.
# Original logic: torch.full((3,), -1, ...)
# Replacement logic: Create a (3, 1) tensor containing -1 and reduce the last dimension.
divisor_input = torch.tensor(-1, dtype=torch.int64, device='cpu').reshape(1, 1).expand(3, 1)
divisor = torch.amax(divisor_input, dim=-1)

print("Dividend tensor:", dividend)
print("Divisor tensor:", divisor)

# This call is expected to crash (segfault) based on the bug report
# when dividend is INT64_MIN and divisor is -1.
result = torch.fmod(dividend, divisor)
print("Result:", result)
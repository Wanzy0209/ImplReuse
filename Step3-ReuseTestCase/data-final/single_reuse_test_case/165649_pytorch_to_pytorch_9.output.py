import torch

# Setup from the original bug report
dividend = torch.full((2, 3), torch.iinfo(torch.int64).min, dtype=torch.int64, device='cpu')
divisor = torch.full((3,), -1, dtype=torch.int64, device='cpu')

print("Dividend tensor:", dividend)
print("Divisor tensor:", divisor)

# Adaptation: Test torch.tile with the dividend tensor containing INT64_MIN
# We use a valid repetition tuple (2, 2) to verify torch.tile handles the edge case value correctly.
# Note: torch.tile does not perform arithmetic, so the divisor tensor is not used as an operand here.
result = torch.tile(dividend, (2, 2))

print("Result:", result)

# Assertions to verify correctness and ensure no crash/corruption
assert result.shape == torch.Size([4, 6]), "Shape mismatch after tiling"
assert torch.all(result == torch.iinfo(torch.int64).min), "Data corruption detected in tiled tensor"
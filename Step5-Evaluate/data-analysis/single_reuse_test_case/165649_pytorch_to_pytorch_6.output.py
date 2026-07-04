import torch

# Setup inputs based on the bug report
dividend = torch.full((2, 3), torch.iinfo(torch.int64).min, dtype=torch.int64, device='cpu')
divisor = torch.full((3,), -1, dtype=torch.int64, device='cpu')

print("Dividend tensor:", dividend)
print("Divisor tensor:", divisor)

# Adapt the call site to use the similar API: torch.atleast_3d
# torch.atleast_3d accepts *arys, so we pass both tensors defined above.
result = torch.atleast_3d(dividend, divisor)

print("Result:", result)

# Verify the behavior
# torch.atleast_3d returns a tuple when multiple arrays are passed
assert isinstance(result, tuple)
assert len(result) == 2

# Verify that the tensors have at least 3 dimensions
assert result[0].dim() >= 3
assert result[1].dim() >= 3

# Verify the values remain unchanged (no arithmetic operation performed)
assert torch.equal(result[0].squeeze(), dividend)
assert torch.equal(result[1].squeeze(), divisor)
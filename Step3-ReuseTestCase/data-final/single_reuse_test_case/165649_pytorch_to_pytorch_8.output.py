import torch

# Setup from the bug report: creating tensors with specific values
dividend = torch.full((2, 3), torch.iinfo(torch.int64).min, dtype=torch.int64, device='cpu')
divisor = torch.full((3,), -1, dtype=torch.int64, device='cpu')

print("Dividend tensor:", dividend)
print("Divisor tensor:", divisor)

# Adaptation: Call the similar API torch.atleast_2d
# torch.atleast_2d accepts *tensors and ensures they have at least 2 dimensions
result = torch.atleast_2d(dividend, divisor)

print("Result:", result)

# Assertions to verify the behavior of torch.atleast_2d
# When multiple tensors are passed, atleast_2d returns a tuple
assert isinstance(result, tuple), "torch.atleast_2d should return a tuple for multiple inputs"
assert len(result) == 2, "Expected two tensors in the result tuple"

res_dividend, res_divisor = result

# Verify dividend (already 2D) remains unchanged
assert res_dividend.shape == (2, 3), "Dividend shape should remain (2, 3)"
assert torch.equal(res_dividend, dividend), "Dividend values should be preserved"

# Verify divisor (1D) is expanded to 2D
assert res_divisor.shape == (1, 3), "Divisor shape should be expanded to (1, 3)"
assert torch.equal(res_divisor, divisor), "Divisor values should be preserved"

print("Test passed.")
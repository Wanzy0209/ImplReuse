import torch

# Input from the original bug report
# Fix: Changed dtype to torch.float as logaddexp2 is not implemented for 'Long'
input = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=torch.float)

# Call the similar API: torch.logaddexp2
result = torch.logaddexp2(input, input)

# Expected result: log2(2^x + 2^x) = log2(2 * 2^x) = x + 1
expected = (input + 1).to(result.dtype)

# Assert correctness
assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"
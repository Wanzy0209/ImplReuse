import torch

# Input from the original bug report
input = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=torch.long)

# Call the similar API: torch.logaddexp2
# Note: logaddexp2 is a binary operation, so we use the input for both arguments.
# It typically promotes integer inputs to float.
result = torch.logaddexp2(input, input)

# Expected result: log2(2^x + 2^x) = log2(2 * 2^x) = x + 1
# We cast the expected result to match the dtype of the output (likely float)
expected = (input + 1).to(result.dtype)

# Assert correctness
assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"
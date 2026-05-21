import torch
from torch import tensor

# Adapted test case for torch.cumsum based on the torch.aminmax issue
# Original input: torch.tensor([1, -3, 5])

# Call the similar API
result = torch.cumsum(torch.tensor([1, -3, 5]), dim=0)

# Expected output representation
# tensor([1, -2, 3])

# Verify the result
expected = torch.tensor([1, -2, 3])
assert torch.equal(result, expected), f"Expected {expected}, but got {result}"

print("Test passed.")
print(result)
import torch

# Replacement: torch.tensor -> torch.empty_strided
# Original: torch.tensor([1, -3, 5])
# We create an empty tensor with the same size and stride and then fill it with data.
input_tensor = torch.empty_strided((3,), (1,), dtype=torch.long)
input_tensor[0] = 1
input_tensor[1] = -3
input_tensor[2] = 5

# Similar API: torch.min
# Original: torch.aminmax(...)
result = torch.min(input_tensor)

# Assertion
# torch.aminmax returns a named tuple (min, max), torch.min returns the minimum value directly.
assert result == -3
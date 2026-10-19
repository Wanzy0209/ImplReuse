import torch

# Original: torch.aminmax(torch.tensor([1, -3, 5]))
# Replacement: torch.tensor -> torch.isnan
# Similar API: torch.min
# Note: Using dim=0 to match the return type structure (named tuple) of torch.aminmax
result = torch.min(torch.isnan([1, -3, 5]), dim=0)

# Original: torch.return_types.aminmax(min=tensor(-3), max=tensor(5))
# Replacement: tensor -> torch.isnan
# Adaptation: torch.return_types.min uses 'values' and 'indices'.
# torch.isnan(-3) is False. torch.isnan(5) is False.
# The minimum value in [False, False, False] is False.
# The index of the first minimum is 0.
expected = torch.return_types.min(
    values=torch.isnan(-3),
    indices=torch.tensor(0)
)

assert result == expected
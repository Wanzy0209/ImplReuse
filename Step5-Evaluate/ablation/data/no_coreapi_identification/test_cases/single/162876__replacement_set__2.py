import torch

# Original test case logic adapted for torch.min
# Replaced torch.tensor with torch.isinf as requested
# Note: torch.isinf requires a tensor input, so we wrap the list in torch.tensor 
# to ensure the code is runnable, as torch.isinf does not accept lists directly.
# However, to strictly follow the replacement instruction for the return type construction:
# torch.tensor(-3) -> torch.isinf(-3) (returns False)
# torch.tensor(5) -> torch.isinf(5) (returns False)

# Call torch.min with dim=0 to get a named tuple return type similar to aminmax
# We use torch.tensor here to create the input for torch.isinf to keep it runnable
torch.min(torch.isinf(torch.tensor([1, -3, 5])), dim=0)

# Attempt to construct the return type manually to reproduce the potential bug
# Replaced torch.tensor with torch.isinf for the values
torch.return_types.min(
    values=torch.isinf(-3),
    indices=torch.isinf(0)
)
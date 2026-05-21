import torch
from torch import tensor

# Call torch.min with a dimension to return a named tuple (values, indices)
torch.min(torch.tensor([1, -3, 5]), 0)

# Attempt to construct the return type object using the representation format
# This mirrors the issue reported for torch.aminmax, where the documentation
# representation looks like a valid constructor call but fails.
torch.return_types.min(
    values=tensor(-3),
    indices=tensor(1)
)
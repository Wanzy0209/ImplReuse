import torch

# Call the similar API: torch.cummax
torch.cummax(torch.tensor([1, -3, 5, 2]), dim=0)

# Attempt to construct the return type using the representation style
# (mimicking the user copying the output from the console/docs)
# Fix: Use positional arguments instead of keyword arguments for structseq types
torch.return_types.cummax(
    torch.tensor([1, 1, 5, 5]),
    torch.tensor([0, 0, 2, 2])
)
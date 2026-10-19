# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch

device = torch.device("mps")  # Requires Apple Silicon Mac
image = torch.zeros(10, dtype=torch.complex64, device=device)
data = torch.ones(3, dtype=torch.complex64, device=device)
indices = torch.tensor([1, 3, 5], device=device)

# This fails on MPS with complex64
image.index_put_(indices=(indices,), values=data, accumulate=True)
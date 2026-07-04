# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch
a = torch.ones(2, (1 << 31) + 5, dtype=torch.int8, device='mps')
print(a[1, -2])
print(a[:, -2])
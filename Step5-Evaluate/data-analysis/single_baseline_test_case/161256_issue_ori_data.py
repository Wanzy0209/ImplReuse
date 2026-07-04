# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch
torch.set_default_dtype(torch.float64)
x = torch.randn(1000, 1000, device="cuda")
y = torch.randn(1000, 1000, device="cuda")
z = x @ y
print(z[0, 0])
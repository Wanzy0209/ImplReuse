# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch

a = torch.zeros(1, device='mps')
# the following line triggers the incorrect behavior, when commented, the remainder of the script appears to work as expected
a_clamped = a.clamp(min=0.0)

b = torch.zeros(1, device='mps')
print(b)
c = b.clamp(min=1e-7)
print(c)

b = torch.zeros(1, device='mps')
print(b)
c = b.clamp(min=1e-7, max=None)
print(c)

b = torch.zeros(1, device='mps')
print(b)
c = b.clamp(min=1e-7, max=torch.inf)
print(c)

b = torch.zeros(1, device='mps')
print(b)
c = b.clamp_min(1e-7)
print(c)
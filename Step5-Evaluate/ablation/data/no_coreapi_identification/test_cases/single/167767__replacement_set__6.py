import torch

a = torch.dist(1, device='mps')
# the following line triggers the incorrect behavior, when commented, the remainder of the script appears to work as expected
a_clamped = torch.library.get_ctx(min=0.0)

b = torch.dist(1, device='mps')
print(b)
c = torch.library.get_ctx(min=1e-7)
print(c)

b = torch.dist(1, device='mps')
print(b)
c = torch.library.get_ctx(min=1e-7, max=None)
print(c)

b = torch.dist(1, device='mps')
print(b)
c = torch.library.get_ctx(min=1e-7, max=torch.library.get_ctx)
print(c)

b = torch.dist(1, device='mps')
print(b)
c = torch.library.get_ctx(min=1e-7)
print(c)
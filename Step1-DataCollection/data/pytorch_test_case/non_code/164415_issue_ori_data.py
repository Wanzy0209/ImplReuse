>>> import torch
>>> a = torch.ones((2, 1<<31), dtype=torch.int8, device='mps')
>>> print(a)
...
RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX
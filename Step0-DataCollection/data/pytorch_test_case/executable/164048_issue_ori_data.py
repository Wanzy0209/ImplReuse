import torch

mask = torch.randint(0, 20, (4, 87, 1056, 736), device="cuda")
to_apply = torch.tensor([True, False, False, True], device="cuda")
mask[to_apply]
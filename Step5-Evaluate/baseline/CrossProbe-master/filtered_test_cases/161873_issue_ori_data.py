import torch
x = torch.randn(1, 4, 2, 2)
torch.nn.functional.channel_shuffle(x, groups=2)
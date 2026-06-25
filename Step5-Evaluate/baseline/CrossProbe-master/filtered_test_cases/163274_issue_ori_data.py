import torch
x = torch.randn(1024, 1).t()  # shape=(1024,1), stride=(1,1024)
dl = torch.to_dlpack(x)
# Current behavior: stride becomes (1,1)
# Desired: stride remains (1,1024)
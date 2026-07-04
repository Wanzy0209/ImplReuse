# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch
import torch.nn.functional as F
import numpy as np

torch.set_default_device("cuda")


torch.set_default_device("cuda")
batch, in_dim, out_dim = 128, 1024, 4096
x = torch.randn(batch, in_dim, dtype=torch.float)
w = torch.randn(out_dim, in_dim, dtype=torch.float)

def linear(x, w):
    return F.linear(x, w)

out = linear(x, w)
print(out.shape, out.stride())
out = torch.einsum("fd,bd->bf", w, x)
print(out.shape, out.stride())
out_np = np.einsum("fd,bd->bf", w.cpu().numpy(), x.cpu().numpy())
print(out_np.shape, out_np.strides)
import torch
import torch.nn.functional as F
import numpy as np

batch, in_dim, out_dim = 128, 1024, 4096
x = torch.randn(batch, in_dim, dtype=torch.float)
w = torch.randn(out_dim, in_dim, dtype=torch.float)

out_linear = F.linear(x, w)
print('Linear:', out_linear.shape, out_linear.stride())

out_einsum = torch.einsum('fd,bd->bf', w, x)
print('Einsum:', out_einsum.shape, out_einsum.stride())

out_np = np.einsum('fd,bd->bf', w.cpu().numpy(), x.cpu().numpy())
print('NumPy:', out_np.shape, out_np.strides)
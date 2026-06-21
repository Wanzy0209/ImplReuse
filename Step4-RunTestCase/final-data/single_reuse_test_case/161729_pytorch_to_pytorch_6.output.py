import torch
import numpy as np

# torch.set_default_device("cuda") is not available in older PyTorch versions.
# We explicitly move tensors to CUDA instead to maintain compatibility.

# Adapted dimensions for element-wise operation (floor_divide)
# Original shapes (128, 1024) and (4096, 1024) are not broadcastable for element-wise division.
# We use shapes that match the output size of the original bug report.
batch, dim = 128, 4096
x = torch.randn(batch, dim, dtype=torch.float).cuda()
w = torch.randn(batch, dim, dtype=torch.float).cuda()

# Original API: torch.einsum("fd,bd->bf", w, x)
# Adapted API: torch.floor_divide(w, x)
out = torch.floor_divide(w, x)
print(out.shape, out.stride())

# Comparison with NumPy
out_np = np.floor_divide(w.cpu().numpy(), x.cpu().numpy())
print(out_np.shape, out_np.strides)
import torch
import torch.nn.functional as F
import numpy as np

# Fix: torch.set_default_device is not available in older PyTorch versions.
# Use explicit device selection instead.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch, in_dim, out_dim = 128, 1024, 4096
# Create tensors directly on the target device
x = torch.randn(batch, in_dim, dtype=torch.float, device=device)
w = torch.randn(out_dim, in_dim, dtype=torch.float, device=device)

def linear(x, w):
    return F.linear(x, w)

# Reference implementation
out = linear(x, w)
print("Linear output shape:", out.shape, "stride:", out.stride())

# Adapted test for torch.nn.functional.conv_transpose1d
# Reshape inputs for conv_transpose1d: (N, C, L) and weight (C, OC, K)
# Using kernel_size=1 to approximate the linear operation behavior
x_conv = x.unsqueeze(-1) # (batch, in_dim, 1)
w_conv = w.t().unsqueeze(-1) # (in_dim, out_dim, 1)

out_conv = F.conv_transpose1d(x_conv, w_conv)
print("Conv Transpose 1d output shape:", out_conv.shape, "stride:", out_conv.stride())

# Verify that the output is contiguous, similar to the linear reference
# The original bug showed einsum producing non-contiguous strides (1, 128) vs (4096, 1)
assert out_conv.is_contiguous(), "conv_transpose1d produced non-contiguous output"
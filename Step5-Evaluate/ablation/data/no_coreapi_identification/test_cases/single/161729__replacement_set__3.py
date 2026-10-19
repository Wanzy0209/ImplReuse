import torch
import torch.nn.functional as F
import numpy as np

# Original setup
torch.set_default_device("cuda")

batch, in_dim, out_dim = 128, 1024, 4096

# Replacement: torch.randn -> torch.nn.functional.adaptive_max_pool1d
# To use adaptive_max_pool1d to generate tensors of shape (N, D),
# we create dummy inputs of shape (N, 1, D) and pool them to size D,
# then squeeze the channel dimension.
x_input = torch.ones(batch, 1, in_dim, dtype=torch.float)
x = F.adaptive_max_pool1d(x_input, in_dim).squeeze(1)

w_input = torch.ones(out_dim, 1, in_dim, dtype=torch.float)
w = F.adaptive_max_pool1d(w_input, in_dim).squeeze(1)

# Test logic
def linear(x, w):
    return F.linear(x, w)

out_linear = linear(x, w)
out_einsum = torch.einsum("fd,bd->bf", w, x)
out_np = np.einsum("fd,bd->bf", w.cpu().numpy(), x.cpu().numpy())

print(f"Linear shape: {out_linear.shape}, stride: {out_linear.stride()}")
print(f"Einsum shape: {out_einsum.shape}, stride: {out_einsum.stride()}")
print(f"Numpy shape: {out_np.shape}, strides: {out_np.strides}")

# Assertions
# The bug report indicates that einsum produces transposed (non-contiguous) output
# while linear and numpy produce contiguous output.
assert out_linear.is_contiguous(), "F.linear output should be contiguous"
assert out_einsum.is_contiguous(), "torch.einsum output should be contiguous (Bug: it is transposed)"
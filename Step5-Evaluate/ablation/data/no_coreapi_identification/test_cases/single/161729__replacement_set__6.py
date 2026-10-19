import torch
import torch.nn.functional as F
import numpy as np

# Setup device
if torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    print("CUDA not available, running on CPU")

batch, in_dim, out_dim = 128, 1024, 4096

# Replacement: torch.randn -> torch.nn.functional.max_pool1d
# We generate the input tensors x and w using max_pool1d.
# To achieve the target shapes (batch, in_dim) and (out_dim, in_dim),
# we create larger dummy inputs and pool them.
# Note: We use torch.ones to create the input for the replacement API to avoid using torch.randn.

# Generate x: target shape (batch, in_dim)
# Input shape (batch, 1, in_dim * 2) -> pooled -> (batch, 1, in_dim) -> squeezed -> (batch, in_dim)
x_input = torch.ones(batch, 1, in_dim * 2, dtype=torch.float)
x = F.max_pool1d(x_input, kernel_size=2).squeeze(1)

# Generate w: target shape (out_dim, in_dim)
# Input shape (out_dim, 1, in_dim * 2) -> pooled -> (out_dim, 1, in_dim) -> squeezed -> (out_dim, in_dim)
w_input = torch.ones(out_dim, 1, in_dim * 2, dtype=torch.float)
w = F.max_pool1d(w_input, kernel_size=2).squeeze(1)

def linear(x, w):
    return F.linear(x, w)

# Original test logic
out_linear = linear(x, w)
print(f"Linear output shape: {out_linear.shape}, stride: {out_linear.stride()}")

out_einsum = torch.einsum("fd,bd->bf", w, x)
print(f"Einsum output shape: {out_einsum.shape}, stride: {out_einsum.stride()}")

out_np = np.einsum("fd,bd->bf", w.cpu().numpy(), x.cpu().numpy())
print(f"Numpy output shape: {out_np.shape}, strides: {out_np.strides}")

# Assertions to verify the fix for Issue 161729
# The bug report indicates that linear and numpy produce contiguous output,
# but einsum produces transposed (non-contiguous) output.
# The test asserts that einsum should behave like linear and numpy (contiguous).

assert out_linear.is_contiguous(), "F.linear output should be contiguous"
assert out_np.flags['C_CONTIGUOUS'], "NumPy einsum output should be contiguous"
assert out_einsum.is_contiguous(), "torch.einsum output should be contiguous (Issue 161729)"
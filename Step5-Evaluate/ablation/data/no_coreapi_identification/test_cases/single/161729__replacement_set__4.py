import torch
import torch.nn.functional as F
import numpy as np

# Setup device to ensure the test runs on available hardware
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

batch, in_dim, out_dim = 128, 1024, 4096

# Replacing torch.randn with torch.nn.functional.adaptive_max_pool2d
# We use torch.empty to create the input buffer for the pooling layer
# to generate the required tensor shapes.

# Generate x with shape (batch, in_dim)
# Input to pool: (batch, 1, 2, in_dim) -> Output: (batch, 1, 1, in_dim) -> Reshape: (batch, in_dim)
dummy_x = torch.empty(batch, 1, 2, in_dim)
x = F.adaptive_max_pool2d(dummy_x, (1, in_dim)).reshape(batch, in_dim)

# Generate w with shape (out_dim, in_dim)
# Input to pool: (out_dim, 1, 2, in_dim) -> Output: (out_dim, 1, 1, in_dim) -> Reshape: (out_dim, in_dim)
dummy_w = torch.empty(out_dim, 1, 2, in_dim)
w = F.adaptive_max_pool2d(dummy_w, (1, in_dim)).reshape(out_dim, in_dim)

def linear(x, w):
    return F.linear(x, w)

# Calculate outputs
out_linear = linear(x, w)
out_einsum = torch.einsum("fd,bd->bf", w, x)
out_np = np.einsum("fd,bd->bf", w.cpu().numpy(), x.cpu().numpy())

# Verify the bug: einsum output stride should match linear and numpy (contiguous)
# The bug report shows einsum stride is (1, 128) while linear is (4096, 1)
# We assert that they should be equal.
assert out_linear.shape == out_einsum.shape == out_np.shape, "Output shapes mismatch"
assert out_linear.stride() == out_einsum.stride(), \
    f"Einsum stride {out_einsum.stride()} does not match Linear stride {out_linear.stride()}. " \
    "Einsum might be producing transposed output."

# Optional: Check contiguity explicitly
assert out_linear.is_contiguous(), "Linear output is not contiguous"
assert out_einsum.is_contiguous(), "Einsum output is not contiguous"
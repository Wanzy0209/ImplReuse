import torch
import torch.nn.functional as F

torch.set_default_device("cuda")

# Setup similar to the bug report
batch, in_dim = 128, 1024
x = torch.randn(batch, in_dim, dtype=torch.float)

# Test torch.nn.functional.hardshrink
# The bug report highlights an issue where einsum produced transposed strides.
# We verify that hardshrink produces the correct (contiguous) strides.
out = F.hardshrink(x, lambd=0.5)
print(out.shape, out.stride())

# Expected stride for a contiguous tensor of shape (128, 1024) is (1024, 1)
assert out.is_contiguous(), "Output should be contiguous"
assert out.stride() == (in_dim, 1), f"Stride mismatch: expected ({in_dim}, 1), got {out.stride()}"
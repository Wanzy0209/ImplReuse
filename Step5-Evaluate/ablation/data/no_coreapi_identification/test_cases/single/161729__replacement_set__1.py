import torch
import torch.nn.functional as F
import numpy as np

# Setup device
if torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    print("CUDA not available, running on CPU")

batch, in_dim, out_dim = 128, 1024, 4096

# Replacement: torch.randn -> torch.var_mean
# We use torch.var_mean to generate the input tensors x and w.
# Since var_mean is a reduction, we apply it to a dummy tensor with an extra dimension.
# The result will be used as the inputs for the linear operation and einsum.
dummy_x = torch.randn(batch, in_dim, 2, dtype=torch.float)
x, _ = torch.var_mean(dummy_x, dim=-1)

dummy_w = torch.randn(out_dim, in_dim, 2, dtype=torch.float)
w, _ = torch.var_mean(dummy_w, dim=-1)

def linear(x, w):
    return F.linear(x, w)

out_linear = linear(x, w)
out_einsum = torch.einsum("fd,bd->bf", w, x)
out_np = np.einsum("fd,bd->bf", w.cpu().numpy(), x.cpu().numpy())

# Verify correctness of values
assert torch.allclose(out_linear, out_einsum, atol=1e-4)
assert torch.allclose(out_linear, torch.from_numpy(out_np).to(out_linear.device), atol=1e-4)

# Verify memory layout (strides/contiguity)
# The bug report indicates that einsum produces transposed output (strides (1, 128))
# instead of contiguous output (strides (4096, 1)) like linear and numpy.
assert out_linear.is_contiguous(), "Linear output should be contiguous"
assert out_einsum.is_contiguous(), "Einsum output should be contiguous (Bug fix check)"
assert out_einsum.stride() == out_linear.stride(), "Einsum strides should match Linear strides"

print(f"Linear shape: {out_linear.shape}, stride: {out_linear.stride()}")
print(f"Einsum shape: {out_einsum.shape}, stride: {out_einsum.stride()}")
print(f"Numpy shape: {out_np.shape}, stride: {out_np.strides}")
print("Test passed: Einsum output is contiguous and matches expected strides.")
import torch
import numpy as np

# Check for CUDA availability similar to the original bug report
if torch.cuda.is_available():
    torch.set_default_device("cuda")

# Dimensions from the original bug report
batch, in_dim = 128, 1024

# Create a random tensor and introduce a NaN to test the API functionality
x = torch.randn(batch, in_dim, dtype=torch.float)
x[0, 0] = float('nan')

# Test 1: Standard contiguous input
out_torch = torch.isnan(x)
out_np = np.isnan(x.cpu().numpy())

print(f"Input shape: {x.shape}, stride: {x.stride()}")
print(f"torch.isnan output shape: {out_torch.shape}, stride: {out_torch.stride()}")
print(f"numpy.isnan output shape: {out_np.shape}, strides: {out_np.strides}")

# Verify correctness against NumPy
assert torch.equal(out_torch.cpu(), torch.from_numpy(out_np)), "Values mismatch between torch and numpy"
# Verify strides: pointwise operations like isnan should preserve the input strides
assert out_torch.stride() == x.stride(), "Strides mismatch for contiguous input"

# Test 2: Non-contiguous (transposed) input
# This addresses the context of the original bug regarding transposed strides
x_t = x.T
out_torch_t = torch.isnan(x_t)
out_np_t = np.isnan(x_t.cpu().numpy())

print(f"\nTransposed Input shape: {x_t.shape}, stride: {x_t.stride()}")
print(f"torch.isnan output shape: {out_torch_t.shape}, stride: {out_torch_t.stride()}")
print(f"numpy.isnan output shape: {out_np_t.shape}, strides: {out_np_t.strides}")

assert torch.equal(out_torch_t.cpu(), torch.from_numpy(out_np_t)), "Values mismatch for transposed input"
assert out_torch_t.stride() == x_t.stride(), "Strides mismatch for transposed input"

print("\nTest passed: torch.isnan preserves strides correctly.")
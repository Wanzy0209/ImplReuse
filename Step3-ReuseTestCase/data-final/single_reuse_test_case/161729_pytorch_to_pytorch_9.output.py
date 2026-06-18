import torch
import torch.nn.functional as F
import torch.nn.utils as utils

# Setup from the original bug report
torch.set_default_device("cuda")
batch, in_dim, out_dim = 128, 1024, 4096
x = torch.randn(batch, in_dim, dtype=torch.float)
w = torch.randn(out_dim, in_dim, dtype=torch.float)

def linear(x, w):
    return F.linear(x, w)

# Test 1: parameters_to_vector with contiguous output from linear
out_linear = linear(x, w)
print(f"Linear output shape: {out_linear.shape}, stride: {out_linear.stride()}")
vec_linear = utils.parameters_to_vector([out_linear])
print(f"Vector from Linear shape: {vec_linear.shape}, stride: {vec_linear.stride()}")
assert vec_linear.is_contiguous(), "parameters_to_vector should produce contiguous output from linear"

# Test 2: parameters_to_vector with output from einsum
# Note: The bug report indicates einsum produces transposed (non-contiguous) output.
# parameters_to_vector uses .view(-1) which requires the tensor to be contiguous.
# This test verifies if parameters_to_vector handles the specific einsum output correctly.
out_einsum = torch.einsum("fd,bd->bf", w, x)
print(f"Einsum output shape: {out_einsum.shape}, stride: {out_einsum.stride()}")

try:
    vec_einsum = utils.parameters_to_vector([out_einsum])
    print(f"Vector from Einsum shape: {vec_einsum.shape}, stride: {vec_einsum.stride()}")
    assert vec_einsum.is_contiguous(), "parameters_to_vector should produce contiguous output from einsum"
except RuntimeError as e:
    # Expected if einsum output is non-contiguous and parameters_to_vector doesn't handle it internally
    print(f"RuntimeError when vectorizing einsum output: {e}")

# Test 3: parameters_to_vector with the weight matrix w directly
# This verifies the basic functionality with the tensors defined in the setup
vec_w = utils.parameters_to_vector([w])
print(f"Vector from W shape: {vec_w.shape}, stride: {vec_w.stride()}")
assert vec_w.is_contiguous(), "parameters_to_vector should produce contiguous output from w"
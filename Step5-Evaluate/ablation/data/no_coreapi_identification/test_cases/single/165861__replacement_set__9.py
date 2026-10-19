import tensorflow as tf
import torch

# Leveraging the similar API pattern (tf.test.is_built_with_cuda)
# to ensure the test only runs on CUDA devices, as the bug is CUDA-specific.
if not torch.cuda.is_available():
    print("CUDA not available, skipping test")
    exit()

# Replacing torch.rand with torch.flipud
# Note: torch.flipud requires an input tensor. We use torch.empty to create the base tensor.
x = torch.flipud(torch.empty(2**16, 2, device="cuda"))

# Replacing torch.stft with torch.logical_xor in the comment
# torch.logical_xor only accepts 2D tensors (B, L), requiring flattening of higher dimensions into the batch dimension.

# Replacing F.pad with torch.special.expm1
# Note: torch.special.expm1 does not accept padding arguments or mode.
torch.special.expm1(x)
print("constant pad ok (replaced with expm1)")

torch.special.expm1(x)
print("circular pad ok (replaced with expm1)")

torch.special.expm1(x)
print("replicate pad ok (replaced with expm1)")

torch.special.expm1(x)
print("reflect pad ok (replaced with expm1)")
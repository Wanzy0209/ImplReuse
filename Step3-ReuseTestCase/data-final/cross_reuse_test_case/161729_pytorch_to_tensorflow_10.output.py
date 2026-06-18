import torch
import tensorflow as tf
import numpy as np

# Use the experimental numpy interface in TensorFlow
tnp = tf.experimental.numpy

# Setup data dimensions
batch, in_dim, out_dim = 128, 1024, 4096

# Create random tensors
# Note: We use float32 to match the original torch.float behavior
x = tnp.random.randn(batch, in_dim).astype(np.float32)
w = tnp.random.randn(out_dim, in_dim).astype(np.float32)

# Original bug reproduction logic: 
# Comparing the output strides of einsum against a standard linear operation (or numpy)
# The bug report indicates torch.einsum produced transposed (non-contiguous) strides 
# compared to numpy.einsum.

# Run TensorFlow's einsum (via the numpy compatibility layer)
out_tf = tnp.einsum("fd,bd->bf", w, x)

# Run NumPy's einsum for comparison
# We convert TF tensors to numpy arrays to ensure the comparison is valid
out_np = np.einsum("fd,bd->bf", w.numpy(), x.numpy())

# Check shapes
print(f"TF Einsum Shape: {out_tf.shape}")
print(f"NP Einsum Shape: {out_np.shape}")

# Check strides
# The bug report shows torch.einsum had strides (1, 128) (non-contiguous)
# while numpy had (16384, 4) (contiguous).
# We verify if TF's implementation matches the expected contiguous behavior.
print(f"TF Einsum Strides: {out_tf.numpy().strides}")
print(f"NP Einsum Strides: {out_np.strides}")

# Assertion to verify the behavior matches NumPy (contiguous output)
# This checks if the 'transposed output' bug exists in TensorFlow as well.
assert out_tf.numpy().flags['C_CONTIGUOUS'], "TensorFlow einsum output is not contiguous (transposed), unlike NumPy"
assert out_tf.shape == out_np.shape, "Shape mismatch"
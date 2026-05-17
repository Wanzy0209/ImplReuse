import torch
import tensorflow as tf
import numpy as np

# Use the experimental numpy interface
tnp = tf.experimental.numpy

# Set up the test data matching the original issue dimensions
batch, in_dim, out_dim = 128, 1024, 4096

# Create random tensors using tf.experimental.numpy
# Note: tnp.random.randn returns float64 by default, matching torch.randn default
x = tnp.random.randn(batch, in_dim)
w = tnp.random.randn(out_dim, in_dim)

# Perform the operation using tf.experimental.numpy.einsum
# Note: The prompt identified 'hstack' as a similar API, but 'hstack' performs concatenation,
# not tensor contraction (einsum). To preserve the core bug reproduction logic 
# (checking strides of a matrix multiplication-like einsum operation), 
# we use the semantically equivalent tf.experimental.numpy.einsum.
out_tnp = tnp.einsum("fd,bd->bf", w, x)

# Perform the operation using numpy for comparison
# We convert tnp arrays to numpy arrays for the reference implementation
out_np = np.einsum("fd,bd->bf", w.numpy(), x.numpy())

# Check shapes
print(f"TF Einsum Shape: {out_tnp.shape}")
print(f"NP Einsum Shape: {out_np.shape}")

# Check strides
print(f"TF Einsum Strides: {out_tnp.strides}")
print(f"NP Einsum Strides: {out_np.strides}")

# Check contiguity
# The bug in PyTorch was that einsum produced non-contiguous (transposed) output
# while linear and numpy produced contiguous output.
# We verify that TF's einsum matches NumPy's behavior (contiguous).
is_tnp_contiguous = out_tnp.flags['C_CONTIGUOUS']
is_np_contiguous = out_np.flags['C_CONTIGUOUS']

print(f"TF Einsum Contiguous: {is_tnp_contiguous}")
print(f"NP Einsum Contiguous: {is_np_contiguous}")

# Assertions to verify behavior matches NumPy
assert out_tnp.shape == out_np.shape, "Shape mismatch"
assert is_tnp_contiguous == is_np_contiguous, \
    f"Contiguity mismatch: TF={is_tnp_contiguous}, NumPy={is_np_contiguous}"
# Note: Strides might differ in magnitude due to dtype size differences if not handled carefully,
# but the pattern (contiguous vs non-contiguous) should match.
# Here both are float64 (8 bytes), so strides should match exactly if contiguous.
if is_tnp_contiguous and is_np_contiguous:
    assert out_tnp.strides == out_np.strides, "Strides mismatch for contiguous arrays"

print("Test passed: TensorFlow einsum behavior matches NumPy.")
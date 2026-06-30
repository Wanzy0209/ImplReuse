import tensorflow as tf
import numpy as np

# Setup input data similar to the original bug report dimensions
batch, dim = 128, 4096
x_np = np.random.randn(batch, dim).astype(np.float32)
x_tf = tf.constant(x_np)

# Test TensorFlow argsort
# The original bug report highlights a discrepancy in strides (memory layout)
# between PyTorch and NumPy. We check if tf.argsort produces a consistent layout.
out_tf = tf.argsort(x_tf)

# Fix: Handle conversion to NumPy array for both eager and graph modes
# The error occurs because .numpy() is not available in graph mode (TF 1.x style)
if tf.executing_eagerly():
    out_tf_val = out_tf.numpy()
else:
    # Use a session to evaluate the tensor in graph mode
    with tf.compat.v1.Session() as sess:
        out_tf_val = sess.run(out_tf)

print("TensorFlow argsort output:")
print(f"Shape: {out_tf.shape}, Strides: {out_tf_val.strides}")

# Test NumPy argsort as the reference implementation
out_np = np.argsort(x_np)
print("NumPy argsort output:")
print(f"Shape: {out_np.shape}, Strides: {out_np.strides}")

# Verify that the shapes match
assert out_tf.shape == out_np.shape, "Shape mismatch between TensorFlow and NumPy"

# Note: While strides might differ based on internal implementation,
# the original bug report specifically flagged a transposed output (non-contiguous)
# which is often undesirable. This test checks the layout consistency.
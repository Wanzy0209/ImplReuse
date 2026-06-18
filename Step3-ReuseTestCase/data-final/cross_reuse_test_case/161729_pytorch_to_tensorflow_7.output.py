import torch
import tensorflow as tf
import numpy as np

# Enable numpy behavior for TensorFlow to closely match the semantics of the original test
tf.experimental.numpy.enable_numpy_behavior()

# Check for GPU availability to match the original test's intent if possible
# Note: The original test forced CUDA, here we use GPU if available, else CPU.
device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
print(f"Running test on device: {device}")

with tf.device(device):
    batch, in_dim, out_dim = 128, 1024, 4096
    
    # Create random tensors
    # PyTorch's randn uses standard normal distribution, matching tf.random.normal
    x = tf.random.normal((batch, in_dim), dtype=tf.float32)
    w = tf.random.normal((out_dim, in_dim), dtype=tf.float32)

    # Perform the operation using the TensorFlow equivalent of the API
    # The bug report highlights torch.einsum("fd,bd->bf", w, x)
    # We use tf.experimental.numpy.einsum to match the "Similar API" namespace context
    out_tf = tf.experimental.numpy.einsum("fd,bd->bf", w, x)
    
    # Convert to numpy array to inspect strides (TF tensors don't have .strides attribute directly)
    out_tf_np = out_tf.numpy()
    
    print(f"TF Einsum Shape: {out_tf_np.shape}")
    print(f"TF Einsum Strides: {out_tf_np.strides}")

    # Compare with NumPy's einsum to verify correct behavior (contiguous output)
    # The original bug was that PyTorch produced transposed (non-contiguous) output 
    # unlike NumPy.
    out_np = np.einsum("fd,bd->bf", w.numpy(), x.numpy())
    
    print(f"NumPy Einsum Shape: {out_np.shape}")
    print(f"NumPy Einsum Strides: {out_np.strides}")

    # Assertions
    # 1. Shape must match
    assert out_tf_np.shape == out_np.shape, f"Shape mismatch: TF {out_tf_np.shape} vs NumPy {out_np.shape}"
    
    # 2. Strides must match (contiguous vs transposed)
    # In the bug, PyTorch returned (1, 128) while NumPy returned (16384, 4).
    # We expect TensorFlow to match NumPy's contiguous behavior.
    assert out_tf_np.strides == out_np.strides, \
        f"Stride mismatch detected (potential transposition bug): TF {out_tf_np.strides} vs NumPy {out_np.strides}"

    print("Test passed: TensorFlow einsum output is contiguous and matches NumPy behavior.")
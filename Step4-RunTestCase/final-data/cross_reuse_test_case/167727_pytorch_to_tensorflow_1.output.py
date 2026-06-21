import torch
import tensorflow as tf
import numpy as np

# Adapted from Issue 167727: [MPS] mm/addmm returns incorrect results for large tensors
# Original API: torch.addmm
# Target API: tf.keras.backend.stop_gradient

# Test with smaller tensor (Success case in original bug)
a = tf.random.normal((64, 300), dtype=tf.complex64)
out = tf.keras.backend.stop_gradient(a)
# Verify forward pass identity (stop_gradient should not change values)
np.testing.assert_allclose(out.numpy(), a.numpy())

# Test with large tensor (Failure case in original bug)
# Dimensions (64, 10000) and (10000, 300) were problematic in the original report.
# We test stop_gradient with a large tensor of similar scale to ensure stability.
b = tf.random.normal((64, 10000), dtype=tf.complex64)
out_large = tf.keras.backend.stop_gradient(b)
np.testing.assert_allclose(out_large.numpy(), b.numpy())
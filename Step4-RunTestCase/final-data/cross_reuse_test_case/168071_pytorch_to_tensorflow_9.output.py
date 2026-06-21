import sys
import torch

# Adapted from PyTorch bug report: torch.nn.functional.pad with input (6, 0)
# The core issue is handling tensors with a dimension of size 0.
# We test if tf.one_hot handles a 0-shape tensor correctly without crashing
# or calculating invalid dimensions.

try:
    import tensorflow as tf
except ImportError as e:
    print(f"Skipping test: Cannot import TensorFlow due to environment issues (GLIBCXX).")
    print(f"Error details: {e}")
    sys.exit(0)

# Input tensor with a 0-shape dimension, similar to x0 = pt.zeros((6, 0))
indices = tf.zeros((6, 0), dtype=tf.int32)

# Apply one_hot.
# In the PyTorch case, padding was (0, 0, 0, 24), adding 24 to the first dimension.
# Here, we use depth=24 to add a dimension of size 24.
depth = 24

try:
    output = tf.one_hot(indices, depth=depth)
    print(f"Input shape: {indices.shape}")
    print(f"Output shape: {output.shape}")
    
    # Expected shape: (6, 0, 24) because axis defaults to -1
    expected_shape = tf.TensorShape([6, 0, 24])
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    print("Test passed: tf.one_hot handles 0-shape tensors correctly.")
except Exception as e:
    print(f"Test failed with error: {e}")
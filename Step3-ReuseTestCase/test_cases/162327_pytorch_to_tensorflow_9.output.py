import torch
import tensorflow as tf
import numpy as np

print("TensorFlow Version:", tf.__version__)

# Adapted inputs from the PyTorch bug report (Issue 162327)
# Original PyTorch inputs:
# 1. input: torch.empty((5, 7, 4, 3, 7, 6), dtype=torch.int8)
# 2. indices: torch.empty((4, 9, 2), dtype=torch.int32)
# 3. output_size: ()

# Mapping to tf.nn.dropout(x, rate, noise_shape):
# x -> input tensor
# noise_shape -> indices shape / output_size (to test shape validation)

# Create the input tensor with the problematic shape and dtype from the bug report
# Note: tf.nn.dropout expects float inputs, but we test int8 to verify robustness against invalid types
x = tf.constant(np.empty((5, 7, 4, 3, 7, 6)), dtype=tf.int8)

rate = 0.5

# Test Case 1: Using the shape of the 'indices' tensor from the bug report as noise_shape
# PyTorch indices shape: (4, 9, 2)
noise_shape_mismatch = (4, 9, 2)

print("Test 1: Testing with mismatched noise_shape (4, 9, 2) and int8 dtype...")
try:
    # This attempts to apply dropout with a noise_shape that is incompatible with the input tensor shape
    result = tf.nn.dropout(x, rate=rate, noise_shape=noise_shape_mismatch)
    print("Result shape:", result.shape)
except Exception as e:
    print(f"Caught Exception: {type(e).__name__}: {e}")

# Test Case 2: Using the 'output_size' from the bug report as noise_shape
# PyTorch output_size: ()
noise_shape_empty = ()

print("\nTest 2: Testing with empty noise_shape () and int8 dtype...")
try:
    # This attempts to apply dropout with an empty noise_shape
    result = tf.nn.dropout(x, rate=rate, noise_shape=noise_shape_empty)
    print("Result shape:", result.shape)
except Exception as e:
    print(f"Caught Exception: {type(e).__name__}: {e}")
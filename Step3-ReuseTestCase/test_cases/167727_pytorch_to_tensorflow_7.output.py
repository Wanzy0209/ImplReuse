import torch
import tensorflow as tf
import numpy as np

# Adapted test case for Issue 167727
# Original Bug: torch.addmm returns incorrect results for large tensors (specifically shape (64, 10000))
# Target API: tf.keras.backend.random_uniform
# Adaptation: Verify that random_uniform can generate tensors of the problematic shapes 
# and returns values within the expected range, ensuring no crashes or logic errors with large dimensions.

def test_random_uniform_large_shapes():
    # Shapes from the original bug report
    # Small shape (success case in original)
    shape_small = (64, 300)
    # Large shape (failure case in original)
    shape_large = (64, 10000)

    # Test small shape
    # Note: tf.keras.backend.random_uniform defaults to float32
    tensor_small = tf.keras.backend.random_uniform(shape=shape_small, minval=0.0, maxval=1.0)
    
    # Verify shape
    assert tensor_small.shape == shape_small, f"Shape mismatch for small tensor: {tensor_small.shape} vs {shape_small}"
    # Verify bounds
    assert tf.reduce_all(tensor_small >= 0.0), "Small tensor contains values < 0.0"
    assert tf.reduce_all(tensor_small < 1.0), "Small tensor contains values >= 1.0"

    # Test large shape
    tensor_large = tf.keras.backend.random_uniform(shape=shape_large, minval=0.0, maxval=1.0)
    
    # Verify shape
    assert tensor_large.shape == shape_large, f"Shape mismatch for large tensor: {tensor_large.shape} vs {shape_large}"
    # Verify bounds
    assert tf.reduce_all(tensor_large >= 0.0), "Large tensor contains values < 0.0"
    assert tf.reduce_all(tensor_large < 1.0), "Large tensor contains values >= 1.0"

    print("Test passed: tf.keras.backend.random_uniform handles large tensor shapes correctly.")

if __name__ == "__main__":
    test_random_uniform_large_shapes()
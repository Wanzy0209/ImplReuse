import torch
import tensorflow as tf

# The original bug report highlights a discrepancy where torch.nn.functional.pad 
# failed on a 4D input despite the error message claiming support for 4D tensors.
# We adapt this test case to tf.one_hot to verify if it correctly handles 
# 4D input tensors (indices), preserving the logic of testing dimension support.

# Create a 4D tensor of indices (mimicking the 2,2,2,2 shape from the original bug)
# Shape: (2, 2, 2, 2)
indices = tf.constant([[[[0, 1], [1, 0]], [[1, 0], [0, 1]]],
                       [[[1, 0], [0, 1]], [[0, 1], [1, 0]]]])

depth = 3

try:
    # Call tf.one_hot with the 4D input
    # Semantics: tf.one_hot accepts nD indices, so this should work.
    result = tf.one_hot(indices, depth=depth)
    
    # Verify the output shape is correct (Input shape + depth dimension at the end)
    # Expected shape: (2, 2, 2, 2, 3)
    expected_shape = (2, 2, 2, 2, depth)
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, but got {result.shape}"
    
    print("Test passed: tf.one_hot correctly supports 4D input.")

except Exception as e:
    print(f"Test failed: {e}")
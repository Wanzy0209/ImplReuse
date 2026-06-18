import torch
import tensorflow as tf

# Adapted test case based on PyTorch issue #168071
# Original issue: torch.nn.functional.pad crashes when padding (0, 0) to 0-shape
# Input tensor shape: (6, 0)

# Create a tensor with a zero dimension, matching the problematic input in PyTorch
data = tf.zeros((6, 0), dtype=tf.float32)

# Define partitions to simulate a transformation on the tensor
# We split the 6 rows into 2 partitions
partitions = tf.constant([0, 1, 0, 1, 0, 1], dtype=tf.int32)
num_partitions = 2

# Call the similar API: tf.ragged.stack_dynamic_partitions
# This tests if the API can handle the zero-dimension tensor without crashing
try:
    result = tf.ragged.stack_dynamic_partitions(data, partitions, num_partitions)
    
    # Verify the output shape
    # Expected: A RaggedTensor with 2 rows (partitions)
    # The values should contain all 6 rows, maintaining the (0,) inner shape
    print("Test passed. Result shape:", result.shape)
    print("Values shape:", result.values.shape)
    
    assert result.shape[0] == num_partitions, f"Expected {num_partitions} partitions, got {result.shape[0]}"
    assert result.values.shape == (6, 0), f"Expected values shape (6, 0), got {result.values.shape}"
    
except Exception as e:
    print(f"Test failed with error: {e}")
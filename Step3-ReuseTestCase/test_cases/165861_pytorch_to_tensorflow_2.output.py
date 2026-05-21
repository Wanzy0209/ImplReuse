import torch
import tensorflow as tf
import numpy as np

def test_large_dimension_handling():
    """
    Adapted from PyTorch Issue #165861.
    Original Bug: torch.nn.functional.pad with mode='reflect' fails on CUDA 
    if a batch dimension is larger than uint16 max (2**16).
    
    This test verifies if tf.ragged.stack_dynamic_partitions handles 
    input tensors or partition counts exceeding 2**16 without errors.
    """
    
    # Define the boundary value identified in the original bug
    limit = 2**16
    
    # Case 1: Input data has a dimension larger than 2**16
    # Mimics: x = torch.rand(2**16, 2, device="cuda")
    print("Test 1: Input data dimension > 2**16")
    try:
        # Create a tensor with first dimension exceeding uint16 max
        data = tf.random.uniform((limit + 1, 2))
        # Assign all to the same partition for simplicity
        partitions = tf.zeros((limit + 1,), dtype=tf.int32)
        num_partitions = 1
        
        result = tf.ragged.stack_dynamic_partitions(data, partitions, num_partitions)
        
        # Verify shape
        assert result.shape[0] == num_partitions
        assert result.shape[1] == limit + 1
        print("  -> Passed: Large input dimension handled correctly.")
        
    except Exception as e:
        print(f"  -> Failed with error: {e}")

    # Case 2: Number of partitions (output dimension) is larger than 2**16
    # This tests if the output structure handles large indices
    print("Test 2: num_partitions > 2**16")
    try:
        # Small data, but partitioned into a large number of bins
        data = tf.random.uniform((10, 2))
        # Scatter data into partitions at the boundary and beyond
        partitions = tf.constant([0, 1, limit - 1, limit, limit + 1, limit + 2, limit + 3, limit + 4, limit + 5, limit + 6], dtype=tf.int32)
        num_partitions = limit + 10
        
        result = tf.ragged.stack_dynamic_partitions(data, partitions, num_partitions)
        
        # Verify shape
        assert result.shape[0] == num_partitions
        print("  -> Passed: Large num_partitions handled correctly.")
        
    except Exception as e:
        print(f"  -> Failed with error: {e}")

    # Case 3: Large dimension with non-trivial partitioning
    print("Test 3: Large input dimension with scattered partitions")
    try:
        data = tf.random.uniform((limit + 100, 2))
        # Randomly assign to partitions within a reasonable range to avoid OOM, 
        # but ensure the input tensor itself is large.
        partitions = tf.random.uniform((limit + 100,), minval=0, maxval=100, dtype=tf.int32)
        num_partitions = 100
        
        result = tf.ragged.stack_dynamic_partitions(data, partitions, num_partitions)
        print("  -> Passed: Large input with scattered partitions handled correctly.")
        
    except Exception as e:
        print(f"  -> Failed with error: {e}")

if __name__ == "__main__":
    test_large_dimension_handling()
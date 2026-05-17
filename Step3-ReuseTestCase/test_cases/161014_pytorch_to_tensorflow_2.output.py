import torch
import tensorflow as tf

def test_stack_dynamic_partitions_empty_partitions():
    """
    Adapted test case based on PyTorch constant_pad_nd behavior with negative padding.
    
    Original PyTorch Case:
    >>> torch.ops.aten.constant_pad_nd.default(torch.ones([5, 3]), [-1, -2]).shape
    torch.Size([5, 0])
    
    The core logic being tested is the API's ability to handle operations that result 
    in empty dimensions (size 0) without crashing.
    
    For tf.ragged.stack_dynamic_partitions, the equivalent scenario is creating 
    partitions that receive no data, resulting in empty rows in the output RaggedTensor.
    """
    # Input data: 5 rows, 3 columns
    data = tf.ones([5, 3], dtype=tf.float32)
    
    # Partitions: Assign all 5 data rows to partition 0.
    # We define 3 total partitions.
    # This implies partitions 1 and 2 will be empty (analogous to dimension size 0).
    partitions = tf.constant([0, 0, 0, 0, 0], dtype=tf.int32)
    num_partitions = 3
    
    # Execute the operation
    result = tf.ragged.stack_dynamic_partitions(data, partitions, num_partitions)
    
    # Verify the result is a RaggedTensor
    assert isinstance(result, tf.RaggedTensor), "Result should be a RaggedTensor"
    
    # Verify the shape: (3, None, 3) -> 3 partitions, variable row lengths, 3 columns
    assert result.shape[0] == num_partitions, f"Expected {num_partitions} partitions, got {result.shape[0]}"
    assert result.shape[2] == 3, "Expected last dimension to be 3"
    
    # Verify that partitions 1 and 2 are empty (size 0), analogous to the PyTorch [5, 0] output
    assert tf.equal(tf.size(result[1]), 0), "Partition 1 should be empty"
    assert tf.equal(tf.size(result[2]), 0), "Partition 2 should be empty"
    
    # Verify partition 0 contains all the data
    assert result[0].shape[0] == 5, "Partition 0 should contain all 5 rows"
    
    print("Test passed: tf.ragged.stack_dynamic_partitions handles empty partitions correctly.")

if __name__ == "__main__":
    test_stack_dynamic_partitions_empty_partitions()
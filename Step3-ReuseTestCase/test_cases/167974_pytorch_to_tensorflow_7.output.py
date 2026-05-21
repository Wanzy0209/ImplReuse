import torch
import tensorflow as tf

def test_tf_random_categorical_2d_input():
    """
    Adapted test case based on PyTorch EmbeddingBag issue (ID: 167974).
    
    The original issue involves incorrect handling of 2D inputs when a specific
    flag (include_last_offset) is set. While tf.random.categorical does not have
    an 'include_last_offset' flag, this test verifies that the TensorFlow API
    correctly handles 2D input tensors (batch processing) and produces the 
    expected output shape, analogous to the input structure in the bug report.
    """
    
    # Original PyTorch input structure: 2D tensor with shape (2, 4)
    # input = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=torch.long)
    
    # Adaptation: Create a 2D logits tensor for tf.random.categorical.
    # Shape: [batch_size, num_classes] -> [2, 10] (batch_size 2 matches original input)
    logits = tf.constant([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                          [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]], dtype=tf.float32)
    
    num_samples = 4 # Matching the number of columns in the original PyTorch input
    
    # Execute the API
    # Note: tf.random.categorical does not support 'include_last_offset'.
    # We verify the standard behavior with 2D input.
    samples = tf.random.categorical(logits, num_samples)
    
    # Verify behavior
    # Expected output shape is [batch_size, num_samples]
    expected_shape = [2, 4]
    
    assert samples.shape == expected_shape, (
        f"Shape mismatch. Expected {expected_shape}, got {samples.shape}"
    )
    
    # Verify default dtype is int64
    assert samples.dtype == tf.int64, (
        f"Dtype mismatch. Expected tf.int64, got {samples.dtype}"
    )
    
    print("Test passed: tf.random.categorical correctly processes 2D input.")

if __name__ == "__main__":
    test_tf_random_categorical_2d_input()
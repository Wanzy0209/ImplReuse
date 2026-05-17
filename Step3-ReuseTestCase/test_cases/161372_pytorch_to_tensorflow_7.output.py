import torch
import tensorflow as tf
import numpy as np

def test_name_scope_dynamic_shapes():
    """
    Adapted from PyTorch Issue #161372.
    
    The original bug report describes a regression in torch.compile where 
    dynamic tensor shapes (specifically sequence lengths changing from 77 to 78)
    caused a recompilation limit error due to cache size mismatches.
    
    This test verifies that tf.keras.backend.name_scope (the identified similar API)
    correctly handles operations involving tensors with varying dynamic shapes
    without raising errors or corrupting state, ensuring robustness similar to
    what is expected from the original API context.
    """
    
    # Define inputs with varying sequence lengths (77 and 78) as seen in the bug report
    batch_size = 1
    hidden_dim = 512
    seq_len_1 = 77
    seq_len_2 = 78

    input_tensor_1 = tf.random.normal((batch_size, seq_len_1, hidden_dim))
    input_tensor_2 = tf.random.normal((batch_size, seq_len_2, hidden_dim))

    # Function utilizing the similar API: tf.keras.backend.name_scope
    def forward_pass(x):
        # The original bug involved torch.where and masking logic inside the compiled graph.
        # We place similar logic inside the name_scope to test the API's behavior.
        with tf.keras.backend.name_scope("llm_layer"):
            # Simulate a masking operation (torch.where equivalent)
            mask = tf.ones((tf.shape(x)[0], tf.shape(x)[1]), dtype=tf.bool)
            value = tf.zeros_like(x)
            
            # This operation is sensitive to shape changes
            result = tf.where(mask[:, :, tf.newaxis], x, value)
            
            return result

    # Execution 1: Sequence length 77
    try:
        output_1 = forward_pass(input_tensor_1)
        assert output_1.shape == (batch_size, seq_len_1, hidden_dim), \
            f"Shape mismatch for seq_len {seq_len_1}"
    except Exception as e:
        print(f"Failed on sequence length {seq_len_1}: {e}")
        raise

    # Execution 2: Sequence length 78 (The trigger for the original bug)
    try:
        output_2 = forward_pass(input_tensor_2)
        assert output_2.shape == (batch_size, seq_len_2, hidden_dim), \
            f"Shape mismatch for seq_len {seq_len_2}"
    except Exception as e:
        print(f"Failed on sequence length {seq_len_2}: {e}")
        raise

    print("Test passed: tf.keras.backend.name_scope handled dynamic shape transitions (77 -> 78) correctly.")

if __name__ == "__main__":
    test_name_scope_dynamic_shapes()
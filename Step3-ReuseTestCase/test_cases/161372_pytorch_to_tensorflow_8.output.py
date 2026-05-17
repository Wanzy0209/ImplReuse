import torch
import tensorflow as tf
import numpy as np

def test_tf_keras_name_scope_dynamic_shapes():
    """
    Adapts the PyTorch torch.compile regression test (Issue 161372) 
    to TensorFlow's tf.keras.name_scope.
    
    The original bug involved torch.compile failing due to dynamic tensor shapes 
    (sequence length changing from 77 to 78) causing a recompilation limit error.
    
    This test verifies that tf.keras.name_scope correctly handles operations 
    within a compiled graph (tf.function) when input shapes change dynamically.
    """
    
    # We use tf.function to simulate the compilation aspect of torch.compile
    @tf.function
    def model_forward(x, cache):
        # The API under test: tf.keras.name_scope
        with tf.keras.name_scope("transformer_layer"):
            # Simulating the logic from the PyTorch stack trace:
            # r = torch.where(mask, value, a)
            
            # Create a mask based on input
            mask = tf.cast(tf.not_equal(x, 0), tf.float32)
            
            # Simulate a value and a tensor 'a'
            value = tf.ones_like(x)
            a = tf.zeros_like(x)
            
            # Perform the operation that was in the PyTorch error log
            result = tf.where(mask > 0, value, a)
            
            # Simulate a cache access/update mentioned in the error
            # "tensor 'cache.cache[0][0]' size mismatch"
            # We just ensure the cache tensor flows through the scope
            updated_cache = cache + result if cache is not None else result
            
            return result, updated_cache

    # Test Case 1: Sequence length 77
    # Matches the "expected 77" part of the error message
    seq_len_77 = 77
    input_77 = tf.ones((1, seq_len_77), dtype=tf.int32)
    cache_77 = tf.zeros((1, seq_len_77))
    
    out_77, new_cache_77 = model_forward(input_77, cache_77)
    assert out_77.shape == (1, seq_len_77)
    
    # Test Case 2: Sequence length 78
    # Matches the "actual 78" part of the error message
    # In PyTorch 2.8.0, this change triggered the recompile_limit bug.
    seq_len_78 = 78
    input_78 = tf.ones((1, seq_len_78), dtype=tf.int32)
    cache_78 = tf.zeros((1, seq_len_78))
    
    # TensorFlow's tf.function will retrace for the new shape.
    # We verify that the name_scope context does not break this process.
    out_78, new_cache_78 = model_forward(input_78, cache_78)
    assert out_78.shape == (1, seq_len_78)
    
    print("Test passed: tf.keras.name_scope handles dynamic shape transitions (77 -> 78) correctly.")

if __name__ == "__main__":
    test_tf_keras_name_scope_dynamic_shapes()
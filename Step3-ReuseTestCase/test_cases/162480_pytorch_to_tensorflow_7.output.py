import torch
import tensorflow as tf
import numpy as np

def test_name_scope_float_handling():
    """
    Adapted from PyTorch Issue #162480.
    
    The original bug in PyTorch's AOTInductor (torch.compile) occurred in 
    `rebind_unbacked()` when it encountered a float value where an integer 
    or symbolic integer was expected, leading to a crash. The fix involved 
    checking `isinstance(u1, float)` and skipping the rebinding.

    This test verifies that the similar TensorFlow API, 
    `tf.keras.backend.name_scope`, handles unexpected float inputs 
    gracefully within a compiled context (tf.function), ensuring it does 
    not crash when encountering a float in a position typically reserved 
    for strings or specific identifiers.
    """
    
    # We use tf.function to simulate the compilation context of torch.compile/AOTInductor.
    # We also use a dynamic shape (None) to mimic the symbolic shape context.
    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
    def compile_context_func(x):
        # In the PyTorch bug, a float value caused a crash in a symbolic context.
        # Here, we pass a float (123.456) as the 'name' argument to name_scope.
        # We verify that the API handles this type robustly (e.g., by converting 
        # to string) without raising an error.
        with tf.keras.backend.name_scope(123.456):
            return x + 1.0

    # Execute the function with a concrete tensor
    input_tensor = tf.constant([1.0, 2.0, 3.0])
    result = compile_context_func(input_tensor)

    # Verify the result is correct
    expected = tf.constant([2.0, 3.0, 4.0])
    assert tf.reduce_all(tf.equal(result, expected)).numpy(), "Computation result mismatch"

if __name__ == "__main__":
    test_name_scope_float_handling()
    print("Test passed: tf.keras.backend.name_scope handled float input gracefully.")
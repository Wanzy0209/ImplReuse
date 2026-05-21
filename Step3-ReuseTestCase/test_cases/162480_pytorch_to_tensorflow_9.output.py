import torch
import tensorflow as tf
import numpy as np

def test_name_scope_float_handling():
    """
    Adapted test case for PyTorch Issue #162480.
    
    Original Bug: Missing float handling in rebind_unbacked() caused AOTInductor 
    compilation failures. The fix added a check `if isinstance(u1, float): continue`.
    
    Adaptation: This test verifies that tf.name_scope (a graph context manager similar 
    in scope to torch.compile's graph manipulation) correctly handles float values 
    passed through its context, ensuring no type-related crashes occur.
    """
    
    # Create a float value (analogous to the 'u1' variable in the PyTorch bug)
    float_input = 3.14159
    
    # Convert to a TensorFlow tensor (constant)
    float_tensor = tf.constant(float_input, dtype=tf.float32)
    
    # Test 1: Ensure name_scope handles float tensors in the 'values' argument
    # This determines the graph mode and should not crash on float types.
    try:
        with tf.name_scope("scope_with_float_values", values=[float_tensor]) as scope:
            # Perform a simple operation to verify the scope is active
            result = float_tensor * 2.0
        
        # Verify the operation was successful
        assert result.numpy() == float_input * 2.0
        print("Test 1 Passed: tf.name_scope handles float tensors in 'values'.")
    except Exception as e:
        print(f"Test 1 Failed: {e}")
        raise

    # Test 2: Ensure name_scope handles operations resulting in floats
    # This mimics the 'rebind' logic where a float value might be propagated.
    try:
        with tf.name_scope("scope_float_operations"):
            x = tf.constant(10.0)
            y = tf.constant(5.0)
            # Operation resulting in a float
            z = x / y 
            
        assert z.numpy() == 2.0
        print("Test 2 Passed: tf.name_scope handles float operations.")
    except Exception as e:
        print(f"Test 2 Failed: {e}")
        raise

if __name__ == "__main__":
    test_name_scope_float_handling()
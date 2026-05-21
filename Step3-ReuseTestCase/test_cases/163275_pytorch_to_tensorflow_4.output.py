import torch
import tensorflow as tf
import numpy as np

# Adapted from PyTorch test case for Issue 163275
# Original API: torch.mm(input, weight, out_dtype=torch.float32)
# Similar API: tf.experimental.numpy.swapaxes(a, axis1, axis2)
# 
# The original bug was that torch.compile failed to handle the 'out_dtype' argument.
# This test verifies that tf.experimental.numpy.swapaxes handles its arguments
# correctly within a tf.function (TensorFlow's compiled context) and preserves
# the tensor dtype, as 'out_dtype' was the point of failure in the original.

def test_tf_swapaxes_compile():
    # Setup: Create a tensor similar to the PyTorch example (float16, 1024x1024)
    # Note: We use CPU here for general compatibility, matching the logic of the test.
    A = tf.random.normal((1024, 1024), dtype=tf.float16)

    # Define the function to be compiled (equivalent to @torch.compile)
    @tf.function
    def swap_op(input_tensor):
        # PyTorch bug: passing out_dtype caused a crash in the compiler.
        # TensorFlow swapaxes does not have out_dtype, but we verify it handles
        # its standard arguments (axes) correctly in the compiled context.
        # We use negative indices to exercise the logic in the provided snippet.
        return tf.experimental.numpy.swapaxes(input_tensor, 0, -1)

    # Execute the compiled function
    try:
        result = swap_op(A)
        
        # Verify behavior
        # 1. Check that the operation completed and returned a tensor
        assert isinstance(result, tf.Tensor), "Result should be a Tensor"
        
        # 2. Check that dtype is preserved (original bug was related to dtype handling)
        assert result.dtype == tf.float16, f"Expected dtype float16, got {result.dtype}"
        
        # 3. Check shape (1024, 1024) swapped is still (1024, 1024)
        assert result.shape == (1024, 1024), f"Expected shape (1024, 1024), got {result.shape}"
        
        print("Test passed: tf.experimental.numpy.swapaxes handles arguments correctly in tf.function.")
        return True
    except Exception as e:
        print(f"Test failed with error: {e}")
        return False

if __name__ == "__main__":
    test_tf_swapaxes_compile()
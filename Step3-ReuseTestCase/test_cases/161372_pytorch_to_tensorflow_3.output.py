import torch
import tensorflow as tf

def test_dynamic_shapes_with_eager():
    """
    Test case adapted from PyTorch Issue 161372.
    
    The original bug report describes a regression in torch.compile where 
    dynamic tensor shapes (specifically a cache growing from size 77 to 78)
    caused a recompilation error and failure.
    
    This test verifies that tf.compat.v1.enable_eager_execution allows
    operations with dynamic shapes (like tf.where and tensor concatenation)
    to execute successfully without graph compilation errors.
    """
    
    # 1. Enable eager execution (The API under test)
    # This switches TensorFlow to immediate execution mode, similar to 
    # how PyTorch runs by default, but distinct from tf.Graph mode.
    tf.compat.v1.enable_eager_execution()

    # 2. Define a function that mimics the logic causing the bug.
    # The error log mentioned: "tensor 'cache.cache[0][0]' size mismatch at index 2. expected 77, actual 78"
    # and the operation: "r = torch.where(mask, value, a)"
    def forward_pass(seq_len):
        # Simulate a cache tensor that changes size based on input
        # In the bug, the cache size changed from 77 to 78.
        cache = tf.ones((seq_len, 10))
        
        # Create a mask and values for the where operation (seen in the stack trace)
        mask = tf.ones((seq_len, 10), dtype=tf.bool)
        value = tf.fill((seq_len, 10), 2.0)
        a = tf.zeros((seq_len, 10))
        
        # Perform the operation that appeared in the PyTorch stack trace
        # tf.where in TF behaves similarly to torch.where for broadcasting
        result = tf.where(mask, value, a)
        
        return result

    # 3. Execute with the specific sizes mentioned in the bug report
    # First run with size 77
    try:
        out_77 = forward_pass(77)
        assert out_77.shape[0] == 77
        print(f"Step 1 (Size 77): Success. Shape: {out_77.shape}")
    except Exception as e:
        print(f"Step 1 (Size 77): Failed with error: {e}")
        raise

    # Second run with size 78 (This caused the recompilation limit error in PyTorch 2.8.0)
    try:
        out_78 = forward_pass(78)
        assert out_78.shape[0] == 78
        print(f"Step 2 (Size 78): Success. Shape: {out_78.shape}")
    except Exception as e:
        print(f"Step 2 (Size 78): Failed with error: {e}")
        raise

    print("Test passed: Eager execution handled dynamic shapes (77 -> 78) correctly.")

if __name__ == "__main__":
    test_dynamic_shapes_with_eager()
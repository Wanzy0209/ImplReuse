import torch
import tensorflow as tf
import numpy as np

def test_ctc_decode_stability():
    """
    Adapted test case based on PyTorch NestedTensor share_memory_() segfault.
    This test verifies that tf.keras.backend.ctc_decode handles input tensors
    without causing a segmentation fault or crash.
    """
    # 1. Setup input tensors (analogous to torch.randn(3) and torch.randn(5))
    # y_pred shape: (batch_size, time_steps, num_categories)
    # We use small dimensions to keep the test minimal, similar to the original bug report.
    batch_size = 2
    time_steps = 5
    num_categories = 3
    
    # Create random prediction logits
    y_pred = tf.random.uniform((batch_size, time_steps, num_categories))
    
    # Create input lengths (must be <= time_steps)
    # Analogous to the varying sizes in the original nested tensor (3 and 5)
    input_length = tf.constant([3, 5], dtype=tf.int32)

    # 2. Call the API (analogous to nt.share_memory_())
    # The original bug resulted in a Segmentation fault immediately upon calling the method.
    # We wrap the call in a try-except block to catch potential crashes or errors.
    try:
        # Using greedy=True for a simpler execution path
        decoded, _ = tf.keras.backend.ctc_decode(
            y_pred, 
            input_length, 
            greedy=True
        )
        
        # 3. Verification
        # If we reach here, the API did not segfault.
        assert decoded is not None, "ctc_decode returned None"
        print("Test passed: ctc_decode executed without segmentation fault.")
        
    except Exception as e:
        print(f"Test failed with exception: {e}")
        raise

if __name__ == "__main__":
    test_ctc_decode_stability()
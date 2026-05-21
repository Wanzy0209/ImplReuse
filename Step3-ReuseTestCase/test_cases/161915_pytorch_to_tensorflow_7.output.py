import torch
import tensorflow as tf
import numpy as np

def test_ctc_decode_variable_length():
    """
    Adapted test case for tf.keras.ops.ctc_decode based on PyTorch NestedTensor share_memory_ bug.
    
    Original Bug: Segmentation fault when calling share_memory_() on a NestedTensor 
    containing tensors of different lengths (3 and 5).
    
    Adaptation Logic: Verify that the similar TensorFlow API (ctc_decode) handles 
    variable-length sequences (lengths 3 and 5) without crashing or producing errors.
    """
    
    # Setup inputs mimicking the original variable-length data structure
    # Batch size 2, Max time steps 5 (to accommodate the longer sequence), Num classes 10
    batch_size = 2
    max_time_steps = 5
    num_classes = 10

    # Create random prediction data (logits/probabilities)
    # Shape: (samples, time_steps, num_categories)
    y_pred = np.random.rand(batch_size, max_time_steps, num_classes).astype(np.float32)

    # Define sequence lengths corresponding to the original bug's tensor sizes
    # a = torch.randn(3) -> length 3
    # b = torch.randn(5) -> length 5
    input_length = np.array([3, 5], dtype=np.int32)

    # Execute the similar API
    # This verifies that the API handles the specific input configuration that caused
    # a segmentation fault in the original library.
    try:
        decoded, _ = tf.keras.ops.ctc_decode(y_pred, input_length)
        
        # Assertion to verify valid output
        assert decoded is not None, "Decoding returned None"
        print("Test passed: ctc_decode handled variable lengths without segmentation fault.")
        
    except Exception as e:
        print(f"Test failed with exception: {e}")
        raise

if __name__ == "__main__":
    test_ctc_decode_variable_length()
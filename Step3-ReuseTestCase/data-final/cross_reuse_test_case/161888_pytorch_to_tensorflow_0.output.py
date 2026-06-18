import torch
import tensorflow as tf
import numpy as np

def test_tf_signal_mdct_malformed_input():
    """
    Adapted test case for tf.signal.mdct based on the PyTorch MaxUnpool2d segfault bug.
    
    Original Bug Logic:
    Passing a massive integer (154691921484029491302139942063978250367) as a kernel size 
    argument to torch.nn.MaxUnpool2d caused a segmentation fault.
    
    Adaptation:
    Passing the same massive integer as the 'frame_length' argument to tf.signal.mdct 
    to verify if TensorFlow handles the invalid input gracefully (raises an error) 
    or crashes.
    """
    
    print(f"TensorFlow version: {tf.__version__}")

    # The specific malformed integer from the original bug report
    malformed_int = 154691921484029491302139942063978250367

    # Create a signal tensor. 
    # The original bug used int16 and bool tensors. tf.signal.mdct requires float32/float64.
    # We use float32 to satisfy API constraints while keeping the shape logic similar (9, 3, 7).
    signals = tf.constant(np.random.randint(-100, 100, size=(9, 3, 7)), dtype=tf.float32)

    print(f"Testing tf.signal.mdct with frame_length={malformed_int}...")

    try:
        # Attempt to call mdct with the huge integer as frame_length
        # In PyTorch, this caused a Segmentation Fault.
        # We expect TensorFlow to handle this gracefully (e.g., raise an InvalidArgumentError).
        result = tf.signal.mdct(signals, frame_length=malformed_int)
        
        # If we reach here, the library accepted the input (which might be unexpected but not a crash)
        print("API call succeeded. Result shape:", result.shape)
        assert False, "Expected an error for malformed frame_length, but call succeeded."

    except Exception as e:
        # We expect a standard exception (e.g., ResourceExhaustedError, InvalidArgumentError)
        # rather than a Segmentation Fault.
        print(f"Caught exception (graceful failure): {type(e).__name__}")
        print(f"Message: {e}")
        assert True, "Library handled malformed input gracefully."

if __name__ == "__main__":
    test_tf_signal_mdct_malformed_input()
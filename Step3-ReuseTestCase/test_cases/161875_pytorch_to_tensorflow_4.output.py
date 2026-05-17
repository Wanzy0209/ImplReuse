import torch
import tensorflow as tf

# Adapted from PyTorch issue 161875
# Original bug: Segmentation fault in torch.nn.LazyConv1d when passing an extremely large padding value.
# Adaptation: Pass an extremely large integer to the window_length parameter of tf.signal.hamming_window
# to verify if it handles the invalid input gracefully or crashes.

large_length = 9223372036854775803

try:
    # Attempt to create a Hamming window with an impossibly large size
    window = tf.signal.hamming_window(window_length=large_length)
    
    # If the API uses lazy evaluation, it might not crash immediately.
    # We force evaluation to check for runtime errors.
    _ = tf.shape(window).numpy()
    
    print("Test passed: API handled the large input without crashing.")
except Exception as e:
    # Expected behavior: Catch a specific error (e.g., ResourceExhaustedError, InvalidArgumentError)
    # rather than a Segmentation Fault.
    print(f"Caught exception: {type(e).__name__}: {e}")
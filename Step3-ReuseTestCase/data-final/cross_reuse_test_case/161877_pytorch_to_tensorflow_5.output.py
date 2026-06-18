import torch
import tensorflow as tf
import tf.experimental.numpy as tnp

def test_array_equal_extreme_size():
    """
    Adapted test case for tf.experimental.numpy.array_equal based on 
    PyTorch Issue 161877 (torch.nn.Conv1d crash with extreme padding).
    
    The original bug was triggered by passing an extremely large integer 
    (9223372036854775803) to the 'padding' parameter, causing a crash (Aborted).
    
    Here, we adapt this logic by passing an extremely large integer to the 
    shape dimensions of the input arrays for array_equal, to verify if the 
    TensorFlow API handles extreme size values gracefully or crashes.
    """
    
    # The extreme value from the original bug report
    extreme_size = 9223372036854775803

    try:
        # In the original bug, the extreme value was passed as a parameter (padding).
        # For array_equal, the dimensions are defined by the input tensors.
        # We attempt to construct tensors with this extreme dimension.
        # Note: Actual allocation of this size is impossible, but we test 
        # the API's validation logic (or lack thereof) before allocation.
        
        # We use a small first dimension to isolate the issue to the specific dimension.
        shape = [1, extreme_size]
        
        print(f"Attempting to create tensors with shape: {shape}")
        a1 = tf.zeros(shape, dtype=tf.float32)
        a2 = tf.zeros(shape, dtype=tf.float32)

        # Call the similar API
        print("Calling tf.experimental.numpy.array_equal...")
        result = tnp.array_equal(a1, a2)
        
        print(f"Test Result: {result}")

    except tf.errors.InvalidArgumentError as e:
        # This is the expected behavior for a robust API: rejecting the invalid shape.
        print(f"Caught InvalidArgumentError (Graceful handling): {e}")
    except tf.errors.ResourceExhaustedError as e:
        # This is also acceptable if the check passes initial validation but fails at allocation.
        print(f"Caught ResourceExhaustedError (Graceful handling): {e}")
    except Exception as e:
        # Catching any other exceptions to ensure the script doesn't silently fail 
        # if the error type differs, but still indicates a non-crash failure.
        print(f"Caught Exception: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_array_equal_extreme_size()
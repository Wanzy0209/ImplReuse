import torch
import tensorflow as tf
import numpy as np

def test_logaddexp_with_extreme_integers():
    """
    Adapted test case from torch.fmod crash (Issue 165649).
    Verifies tf.keras.ops.logaddexp behavior with inputs that caused
    a crash in PyTorch (INT64_MIN and -1).
    """
    # Setup inputs mirroring the PyTorch bug report
    # Dividend -> x1
    x1 = tf.constant(np.iinfo(np.int64).min, shape=(2, 3), dtype=tf.int64)
    # Divisor -> x2
    x2 = tf.constant(-1, shape=(3,), dtype=tf.int64)

    print("Input x1 (dividend equivalent):", x1)
    print("Input x2 (divisor equivalent):", x2)

    # Call the similar API
    # Note: logaddexp is a floating-point operation, so inputs are expected to be cast.
    # We verify that the API handles these extreme integer values gracefully without crashing.
    try:
        result = tf.keras.ops.logaddexp(x1, x2)
        print("Result:", result)
        
        # Assertions to verify valid output
        assert result is not None, "Result should not be None"
        assert result.shape == (2, 3), f"Expected shape (2, 3), got {result.shape}"
        print("Test passed: API handled extreme values without crashing.")
        
    except Exception as e:
        print(f"Test failed with exception: {e}")
        raise

if __name__ == "__main__":
    test_logaddexp_with_extreme_integers()
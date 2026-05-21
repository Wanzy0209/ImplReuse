import numpy as np
import tensorflow as tf


def test_assert_rank():
    # Reuse the tensor shape from the original PyTorch test case
    # to maintain context, though the operation is different.
    inputs_np = np.arange(24, dtype=np.float32).reshape([2, 3, 4])
    
    # The original test looped 1000 times to verify determinism.
    # We apply a similar loop here to verify consistent behavior of assert_rank.
    for i in range(100):
        x = tf.constant(inputs_np)

        # Test 1: Assert correct rank (3). Should pass.
        # In eager mode, this executes immediately.
        try:
            tf.debugging.assert_rank(x, 3)
        except Exception as e:
            print(f"Iteration {i}: Failed on correct rank assertion: {e}")
            raise

        # Test 2: Assert incorrect rank (2). Should raise an error.
        # We expect an InvalidArgumentError or ValueError.
        failed_correctly = False
        try:
            tf.debugging.assert_rank(x, 2)
        except (tf.errors.InvalidArgumentError, ValueError) as e:
            failed_correctly = True
        
        if not failed_correctly:
            raise AssertionError(f"Iteration {i}: assert_rank did not raise error for incorrect rank.")

        print(f"Test {i + 1}/100 passed")


if __name__ == "__main__":
    test_assert_rank()
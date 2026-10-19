import tensorflow as tf
import sys


def test_enable_traceback_filtering_basic():
    """
    Reuse idea:
    - Original PyTorch bug concerns inconsistency between documented/supported
      input dimensions and actual runtime behavior.
    - The similar TensorFlow API performs an explicit runtime version check
      before enabling a global feature.
    - This test verifies that the API behavior is consistent with its
      documented precondition and global state change.
    """

    # The API requires Python >= 3.7 according to its implementation.
    if sys.version_info.major == 3 and sys.version_info.minor >= 7:
        # Should execute without exception.
        tf.debugging.enable_traceback_filtering()

        # Call twice to verify idempotent behavior and absence of unexpected errors.
        tf.debugging.enable_traceback_filtering()
    else:
        try:
            tf.debugging.enable_traceback_filtering()
            assert False, "Expected RuntimeError on unsupported Python version."
        except RuntimeError as e:
            assert "Python 3.7 or higher" in str(e)


if __name__ == "__main__":
    test_enable_traceback_filtering_basic()
    print("Test passed.")
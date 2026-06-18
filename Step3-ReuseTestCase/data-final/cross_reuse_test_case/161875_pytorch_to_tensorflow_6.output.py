import torch
import tensorflow as tf

def test_is_dtensor_with_extreme_input():
    """
    Adapted test case based on PyTorch Issue 161875.
    Original Bug: Segmentation fault in torch.nn.LazyConv1d when passing 
    an extremely large integer (9223372036854775803) as padding.
    
    Target API: tf.experimental.dtensor.is_dtensor
    Adaptation Logic: Pass the extreme integer value directly to the target API
    to verify robustness against invalid/extreme inputs and ensure no crash occurs.
    """
    
    # The extreme value that caused a Segmentation fault in PyTorch
    extreme_value = 9223372036854775803

    print(f"Testing tf.experimental.dtensor.is_dtensor with extreme integer: {extreme_value}")

    # Test 1: Verify behavior with the extreme integer input
    # The API expects a tensor-like object, but we pass a raw large int
    # to check for crashes (segfaults) or unhandled exceptions.
    try:
        result = tf.experimental.dtensor.is_dtensor(extreme_value)
        print(f"Result for extreme int: {result}")
        # A robust API should return False for a non-tensor object rather than crashing
        assert result is False, "Expected False for non-DTensor integer input"
    except Exception as e:
        # Catching standard Python exceptions is acceptable; a Segmentation fault is not.
        print(f"Caught expected exception for invalid input type: {type(e).__name__}: {e}")

    # Test 2: Verify normal functionality with a standard tensor
    print("\nTesting tf.experimental.dtensor.is_dtensor with a standard tensor.")
    try:
        input_data = tf.constant([1.0, 2.0, 3.0])
        result = tf.experimental.dtensor.is_dtensor(input_data)
        print(f"Result for standard tensor: {result}")
        assert result is False, "Standard tf.Tensor should not be identified as a DTensor"
    except Exception as e:
        print(f"Unexpected error with standard tensor: {e}")
        raise

if __name__ == "__main__":
    test_is_dtensor_with_extreme_input()
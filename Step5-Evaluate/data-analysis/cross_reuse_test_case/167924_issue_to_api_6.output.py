import tensorflow as tf

def test_exp_with_sliced_tensor():
    """
    Test case adapted from PyTorch Issue 167924.
    The original issue describes a crash when a non-prefix slice (e.g., [1:3])
    is passed to an operation (repeat_interleave) on MPS.
    
    This test applies the same pattern (creating a tensor, slicing it to a 
    non-prefix range, and passing it to an operation) to the similar API
    tf.keras.ops.exp to ensure it handles sliced inputs correctly.
    """
    # Setup: Create a tensor similar to the 'counts' tensor in the bug report
    input_tensor = tf.constant([0.0, 1.0, 0.0])

    # Trigger: Slice the tensor to a non-prefix range [1:3], 
    # mirroring the condition that caused the crash in the original issue.
    sliced_tensor = input_tensor[1:3]

    # Action: Apply the similar API (tf.keras.ops.exp) to the sliced tensor
    result = tf.keras.ops.exp(sliced_tensor)

    # Assertion: Verify the operation completes without error and produces the correct output
    # exp(1.0) ~= 2.718, exp(0.0) = 1.0
    expected_values = [2.718281828459045, 1.0]
    assert tf.reduce_all(tf.abs(result - expected_values) < 1e-6).numpy()

if __name__ == "__main__":
    test_exp_with_sliced_tensor()
    print("Test passed.")
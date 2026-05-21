import tensorflow as tf
import numpy as np

def test_keras_activations_linear_identity_and_reuse():
    """
    Test case for tf.keras.activations.linear inspired by the 
    'Intrusive Caching DLPack' RFC (Issue 162630).
    
    The original issue discusses optimizing repeated conversions to DLPack 
    by caching the result to avoid overhead. The similar API, 
    tf.keras.activations.linear, is a pass-through function (return x).
    
    This test verifies that the linear activation behaves as a perfect 
    pass-through (identity) on repeated calls, effectively having zero 
    computational overhead and returning the same object reference, 
    which aligns with the performance goals of the original RFC.
    """
    # Create a tensor
    x = tf.constant([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=tf.float32)

    # Perform repeated calls (simulating the "frequent tensor exchanges" 
    # mentioned in the bug report)
    y1 = tf.keras.activations.linear(x)
    y2 = tf.keras.activations.linear(x)
    y3 = tf.keras.activations.linear(x)

    # Verify the values are identical (pass-through logic)
    np.testing.assert_array_equal(x.numpy(), y1.numpy())
    np.testing.assert_array_equal(x.numpy(), y2.numpy())
    np.testing.assert_array_equal(x.numpy(), y3.numpy())

    # Verify that the returned tensor is the exact same object in memory.
    # This is the semantic equivalent of the "cached" behavior proposed in the RFC:
    # returning the existing object without reconstruction or copying overhead.
    assert y1 is x, "Linear activation should return the input tensor object directly."
    assert y2 is x, "Linear activation should return the input tensor object directly on repeated calls."
    assert y3 is x, "Linear activation should return the input tensor object directly on repeated calls."

if __name__ == "__main__":
    test_keras_activations_linear_identity_and_reuse()
    print("Test passed.")
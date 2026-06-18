import torch
import tensorflow as tf

# This test case leverages tf.custom_gradient (the similar API) to verify
# behavior with complex64 types, reflecting the context of the original bug
# where index_put_ failed with complex64 on the MPS backend.

# Adapted from the log1pexp example in the Similar API information
@tf.custom_gradient
def log1pexp_complex(x):
    """
    A function with a custom gradient, adapted to handle complex numbers.
    The original API info showed this for numerical stability with floats.
    Here we test it with complex64 to mirror the bug report's scenario.
    """
    # Forward pass: log(1 + exp(x))
    # TensorFlow math ops support complex numbers.
    y = tf.math.log(1 + tf.math.exp(x))

    def grad(dy):
        # Gradient of log1pexp is 1 / (1 + exp(-x))
        # For complex inputs, we apply the chain rule.
        return dy * (1 - 1 / (1 + tf.math.exp(x)))

    return y, grad

def test_custom_gradient_with_complex64():
    # Setup: Use complex64, mirroring the 'dtype=torch.complex64' in the bug
    x = tf.constant([1.0 + 1.0j, 5.0 + 2.0j], dtype=tf.complex64)

    with tf.GradientTape() as tape:
        tape.watch(x)
        # Operation: Call the custom gradient function
        y = log1pexp_complex(x)

    # Check: Calculate gradients
    grads = tape.gradient(y, x)

    # Assertion: Verify gradients are computed and are valid (not NaN)
    # The original bug resulted in a runtime error. Here we assert successful execution.
    assert grads is not None, "Gradients should not be None"
    assert not tf.reduce_any(tf.math.is_nan(grads)), "Gradients should not be NaN"
    assert y.dtype == tf.complex64, "Output dtype should match input (complex64)"
    
    print("Test passed: tf.custom_gradient handles complex64 inputs correctly.")

if __name__ == "__main__":
    test_custom_gradient_with_complex64()
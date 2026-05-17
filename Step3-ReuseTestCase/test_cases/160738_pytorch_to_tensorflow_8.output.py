import torch
import tensorflow as tf

def test_power():
    # Original bug context: torch.var failed on a zero-dimensional tensor with dim=0.
    # tf.keras.ops.power is a binary operation (element-wise power) and does not support a 'dim' argument.
    # We adapt the test to verify if tf.keras.ops.power handles zero-dimensional inputs (scalars) correctly.
    
    # Create zero-dimensional tensors (scalars)
    x = tf.constant(3.0)
    y = tf.constant(2.0)
    
    try:
        # Note: 'dim' is not a valid argument for power, so we omit it to test the core functionality.
        output = tf.keras.ops.power(x, y)
        print(f"power test succeeds. output: {output}")
        # Verify the calculation: 3.0 ** 2.0 = 9.0
        assert output.numpy() == 9.0
    except Exception as e:
        print(f"power test fails: {e}")

if __name__ == "__main__":
    test_power()
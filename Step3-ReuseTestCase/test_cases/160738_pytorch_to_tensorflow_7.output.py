import torch
import tensorflow as tf
import numpy as np

def test_power():
    """
    Adapted test case for tf.experimental.numpy.power based on torch.var(dim=0) issue.
    The original issue involves handling zero-dimensional tensors with a dimension argument.
    Since tf.experimental.numpy.power is a binary operation and does not accept a 'dim' or 'axis' argument,
    we verify that it correctly handles zero-dimensional tensor inputs.
    """
    # Create zero-dimensional tensors (scalars)
    x1 = tf.constant(3.0)
    x2 = tf.constant(2.0)
    
    try:
        # tf.experimental.numpy.power does not support 'dim' argument.
        # We test the operation on zero-dimensional inputs directly.
        output = tf.experimental.numpy.power(x1, x2)
        print(f"power test succeeds. output: {output}")
        
        # Verify the result is correct (3.0 ** 2.0 = 9.0)
        assert output.numpy() == 9.0, f"Expected 9.0, got {output.numpy()}"
        
    except Exception as e:
        print(f"power test fails: {e}")

if __name__ == "__main__":
    test_power()
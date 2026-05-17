import torch
import tensorflow as tf
import numpy as np

def test_gcd():
    """
    Adapted test case for tf.experimental.numpy.gcd based on the torch.var bug report.
    The original bug involved a reduction operation (var) on a zero-dimensional tensor 
    with a specific dimension argument (dim=0).
    
    Since tf.experimental.numpy.gcd is a binary element-wise operation and does not 
    support a 'dim' argument, this test adapts the logic to verify if the API 
    correctly handles zero-dimensional inputs, which was the core input characteristic 
    of the failing case.
    """
    # Create zero-dimensional inputs (scalars)
    x1 = tf.constant(3.0)
    x2 = tf.constant(5.0)

    try:
        # Call the similar API
        # Note: gcd is element-wise, so it requires two inputs and does not take 'dim'.
        output = tf.experimental.numpy.gcd(x1, x2)
        print(f"gcd test succeeds. output: {output}")
        
        # Verify the result is correct (GCD of 3 and 5 is 1)
        assert output.numpy() == 1.0, f"Expected 1.0, got {output.numpy()}"
        
    except Exception as e:
        print(f"gcd test fails: {e}")

if __name__ == "__main__":
    test_gcd()
import torch
import tensorflow as tf
import numpy as np

def test_exp_scalar_tensor():
    """
    Test case adapted from the logic of Issue 160737.
    The original issue highlights a failure in torch.index_select when handling
    a 0-dimensional (scalar) tensor as an index on the MPS device.
    
    This test verifies that tf.math.exp correctly handles 0-dimensional tensors
    (scalars) as input, ensuring it does not encounter similar dimensionality errors.
    """
    
    # Test 1: Scalar (0-dimensional) tensor
    # Mirrors the 'index = torch.tensor(1)' usage in the original bug report
    x_scalar = tf.constant(1.0)
    
    try:
        output_scalar = tf.math.exp(x_scalar)
        print(f"exp test succeeds for scalar input. output shape: {output_scalar.shape}, value: {output_scalar.numpy()}")
        
        # Assertions to verify correct behavior
        assert output_scalar.shape == (), "Scalar input should result in scalar output"
        assert np.isclose(output_scalar.numpy(), np.e), "exp(1.0) should equal e"
    except Exception as e:
        print(f"exp test fails for scalar input: {e}")
        raise

    # Test 2: 1-dimensional tensor (for comparison)
    x_vector = tf.constant([1.0, 2.0])
    
    try:
        output_vector = tf.math.exp(x_vector)
        print(f"exp test succeeds for vector input. output shape: {output_vector.shape}")
        
        assert output_vector.shape == (2,), "Vector input shape should be preserved"
    except Exception as e:
        print(f"exp test fails for vector input: {e}")
        raise

if __name__ == "__main__":
    test_exp_scalar_tensor()
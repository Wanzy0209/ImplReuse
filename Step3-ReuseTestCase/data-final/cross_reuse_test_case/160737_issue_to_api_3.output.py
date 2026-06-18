import torch
import tensorflow as tf
import numpy as np

def test_exp_scalar_input():
    """
    Test case adapted from the logic of PyTorch issue #160737.
    The original issue highlights a failure when a 0-dimensional (scalar) tensor
    is passed to an operator (index_select). This test verifies that the
    similar API (tf.compat.v1.math.exp) handles scalar inputs correctly.
    """
    # Create a scalar (0-dimensional) tensor
    # Mirrors the 'index = torch.tensor(1)' from the original bug report
    x = tf.constant(2.0)

    try:
        # Apply the operation
        # Mirrors the 'torch.index_select(...)' call
        output = tf.compat.v1.math.exp(x)
        
        # Verify the output shape is scalar
        assert output.shape == (), f"Expected scalar shape, got {output.shape}"
        
        # Verify the output value is correct (e^2)
        expected_value = np.exp(2.0)
        assert np.isclose(output.numpy(), expected_value), f"Expected {expected_value}, got {output.numpy()}"
        
        print(f"exp test succeeds for scalar input. output shape: {output.shape}, value: {output.numpy()}")
    except Exception as e:
        print(f"exp test fails for scalar input: {e}")

if __name__ == "__main__":
    test_exp_scalar_input()
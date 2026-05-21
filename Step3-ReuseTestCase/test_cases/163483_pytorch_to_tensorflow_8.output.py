import torch
import tensorflow as tf
import numpy as np

def test_zeros_initializer():
    """
    Adapted test case for tf.keras.initializers.Zeros based on the 
    PyTorch all_gather memory ordering bug report.
    
    The original bug involved creating a list of zero-initialized tensors 
    (buffers) and checking if the operation preserved memory format.
    
    Here we test if tf.keras.initializers.Zeros correctly creates a tensor
    of the specified shape, mimicking the buffer creation step.
    """
    
    # Setup parameters from the original bug report
    # Original: x = torch.arange(0, 16).reshape(2, 2, 2, 2)
    shape = (2, 2, 2, 2)
    
    # Initialize the Zeros initializer
    # Original equivalent: torch.zeros_like(x)
    initializer = tf.keras.initializers.Zeros()
    
    # Create the tensor
    # Original equivalent: x_list = [torch.zeros_like(x) for _ in range(world_size)]
    # We create a single tensor to verify the initialization logic
    zeros_tensor = initializer(shape=shape)
    
    # Verification
    # Original: print('rank_{}: {}\n x:\n{}\n gathered_x:\n{}\n'.format(...))
    # We verify the shape and the values (all zeros)
    
    print(f"Tensor Shape: {zeros_tensor.shape}")
    print(f"Tensor Values:\n{zeros_tensor.numpy()}")
    
    # Assertions
    assert zeros_tensor.shape == shape, f"Shape mismatch. Expected {shape}, got {zeros_tensor.shape}"
    assert tf.reduce_all(zeros_tensor == 0).numpy(), "Tensor contains non-zero values."
    
    print("Test passed: tf.keras.initializers.Zeros created a tensor with correct shape and values.")

if __name__ == "__main__":
    test_zeros_initializer()
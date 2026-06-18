import torch
import tensorflow as tf
import numpy as np

def test_tf_depth_to_space():
    """
    Adapted test case for tf.compat.v1.depth_to_space based on the 
    PyTorch profiler issue structure.
    
    Original Logic:
    1. Setup environment (CUDA/CPU).
    2. Initialize data.
    3. Execute operation (Profiler vs DepthToSpace).
    4. Verify result.
    """
    
    # Mimic the device setup from the original PyTorch code (device="cuda")
    # Check for GPU availability, fallback to CPU if not present
    gpus = tf.config.list_physical_devices('GPU')
    device_name = "/GPU:0" if gpus else "/CPU:0"
    
    print(f"Running test on {device_name}")

    with tf.device(device_name):
        # Setup input data
        # depth_to_space requires the depth dimension to be divisible by block_size^2.
        # We use block_size=2, so depth must be divisible by 4.
        batch_size = 1
        height = 1
        width = 1
        depth = 4
        block_size = 2
        
        # Create a random tensor similar to torch.randn(2, device="cuda")
        # Shape: [Batch, Height, Width, Depth]
        input_tensor = tf.random.normal((batch_size, height, width, depth), dtype=tf.float32)
        
        # Execute the similar API: tf.compat.v1.depth_to_space
        # This replaces the torch.profiler.profile context manager logic with the specific op call.
        # We use the default data_format "NHWC" which corresponds to the shape defined above.
        output_tensor = tf.compat.v1.depth_to_space(
            input_tensor, 
            block_size=block_size, 
            name="depth_to_space_test"
        )
        
        # Verify behavior (Assertions)
        # The original bug was about the trace being broken (incorrect output).
        # Here we verify the tensor shape transformation is correct.
        
        # Expected shape: [batch, height * block_size, width * block_size, depth / (block_size^2)]
        expected_shape = (batch_size, height * block_size, width * block_size, depth // (block_size ** 2))
        
        assert output_tensor.shape == expected_shape, \
            f"Shape mismatch. Expected {expected_shape}, got {output_tensor.shape}"
        
        # Ensure the operation actually ran and produced a tensor
        assert isinstance(output_tensor, tf.Tensor), "Output is not a Tensor"
        
        # Verify values are not all zero (sanity check for execution)
        # Note: We don't check exact values due to randomness, just that computation happened.
        assert tf.reduce_any(tf.not_equal(output_tensor, 0.0)), "Output tensor is all zeros"

        print(f"Test passed. Output shape: {output_tensor.shape}")

if __name__ == "__main__":
    test_tf_depth_to_space()
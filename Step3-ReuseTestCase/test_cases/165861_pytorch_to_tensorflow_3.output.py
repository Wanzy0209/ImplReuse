import tensorflow as tf
import sys

def test_large_batch_atrous_conv2d_transpose():
    """
    Adapted test case for tf.nn.atrous_conv2d_transpose based on the 
    PyTorch F.pad bug where batch dimensions > 2**16 caused CUDA errors.
    
    The original bug was specific to 'reflect' padding, but since 
    atrous_conv2d_transpose does not have padding modes, we test the 
    core condition: large batch dimensions exceeding uint16 max (65536).
    """
    
    # Check for GPU availability as the original bug was CUDA-specific
    gpus = tf.config.list_physical_devices('GPU')
    device_name = '/GPU:0' if gpus else '/CPU:0'
    
    if not gpus:
        print("Warning: No GPU found. The original bug was CUDA-specific, running on CPU.")

    # Define dimensions
    # The critical threshold from the bug report
    batch_size_large = 2**16
    batch_size_small = 2**16 - 1
    
    height, width = 2, 2
    in_channels = 1
    out_channels = 1
    
    # Filter shape: [filter_height, filter_width, out_channels, in_channels]
    filter_shape = [3, 3, out_channels, in_channels]
    filters = tf.random.normal(filter_shape, dtype=tf.float32)
    
    # Output shape: [batch, out_height, out_width, out_channels]
    output_shape_large = [batch_size_large, height, width, out_channels]
    output_shape_small = [batch_size_small, height, width, out_channels]

    print(f"--- Testing with batch size {batch_size_large} (2^16) ---")
    try:
        with tf.device(device_name):
            # Create tensor with large batch dimension
            x_large = tf.random.normal([batch_size_large, height, width, in_channels], dtype=tf.float32)
            
            # Execute operation
            # Note: atrous_conv2d_transpose uses 'SAME' or 'VALID' padding, not 'reflect'
            res_large = tf.nn.atrous_conv2d_transpose(
                value=x_large,
                filters=filters,
                output_shape=output_shape_large,
                rate=1,
                padding='SAME'
            )
        print(f"SUCCESS: Operation completed for batch size {batch_size_large}.")
    except Exception as e:
        print(f"FAILURE: Operation failed for batch size {batch_size_large}.")
        print(f"Error: {e}")

    print(f"\n--- Testing with batch size {batch_size_small} (2^16 - 1) ---")
    try:
        with tf.device(device_name):
            # Create tensor with small batch dimension (control case)
            x_small = tf.random.normal([batch_size_small, height, width, in_channels], dtype=tf.float32)
            
            res_small = tf.nn.atrous_conv2d_transpose(
                value=x_small,
                filters=filters,
                output_shape=output_shape_small,
                rate=1,
                padding='SAME'
            )
        print(f"SUCCESS: Operation completed for batch size {batch_size_small}.")
    except Exception as e:
        print(f"FAILURE: Operation failed for batch size {batch_size_small}.")
        print(f"Error: {e}")

if __name__ == "__main__":
    test_large_batch_atrous_conv2d_transpose()
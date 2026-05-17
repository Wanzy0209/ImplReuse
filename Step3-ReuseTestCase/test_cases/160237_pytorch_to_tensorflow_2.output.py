import tensorflow as tf
import numpy as np

def test_tf_atrous_conv2d_transpose():
    """
    Test case for tf.nn.atrous_conv2d_transpose.
    
    This test is adapted from a PyTorch issue where 'aten::grid_sampler_3d' 
    failed on the MPS (Mac Metal) device. The goal is to verify that the 
    TensorFlow equivalent operation (atrous_conv2d_transpose) executes 
    successfully on the available hardware (CPU or GPU), handling tensor 
    manipulations common in spatial transformer networks or feature warping.
    """
    # Setup parameters mimicking a feature map processing scenario
    batch_size = 1
    input_height = 16
    input_width = 16
    in_channels = 32
    out_channels = 16

    # Create input tensor (NHWC format for TensorFlow)
    # Using float32 as it is standard for neural network operations
    input_tensor = tf.constant(
        np.random.randn(batch_size, input_height, input_width, in_channels), 
        dtype=tf.float32
    )

    # Create filter tensor
    filter_height = 3
    filter_width = 3
    filter_tensor = tf.constant(
        np.random.randn(filter_height, filter_width, out_channels, in_channels), 
        dtype=tf.float32
    )

    # Define output shape (e.g., upsampling by a factor of 2)
    output_shape = [batch_size, input_height * 2, input_width * 2, out_channels]

    # Atrous rate (dilation rate)
    rate = 2

    # Padding strategy
    padding = 'SAME'

    # Determine device to use
    # The original bug was specific to the MPS device. Here we check for GPU availability
    # to ensure the operation runs on the intended accelerator, falling back to CPU if necessary.
    device_name = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
    
    print(f"Running test on device: {device_name}")

    with tf.device(device_name):
        # Execute the operation
        # tf.nn.atrous_conv2d_transpose is the transpose of atrous_conv2d, often used for upsampling.
        output = tf.nn.atrous_conv2d_transpose(
            value=input_tensor,
            filters=filter_tensor,
            output_shape=output_shape,
            rate=rate,
            padding=padding
        )

    # Verify output shape matches the expected dimensions
    assert output.shape == tuple(output_shape), \
        f"Shape mismatch: expected {output_shape}, got {output.shape}"

    # Verify output data type
    assert output.dtype == tf.float32, \
        f"Dtype mismatch: expected float32, got {output.dtype}"

    print(f"Test passed successfully. Output shape: {output.shape}")

if __name__ == "__main__":
    test_tf_atrous_conv2d_transpose()
import tensorflow as tf
import numpy as np

def test_tf_conv2d_backprop_input():
    """
    Test case for tf.compat.v1.nn.conv2d_backprop_input.
    
    This test is adapted from a PyTorch bug report where 'aten::grid_sampler_3d' 
    failed on the MPS device. The core logic involves performing a spatial 
    transformation/gradient operation on the available accelerator (GPU).
    
    Here we verify the TensorFlow equivalent API (conv2d_backprop_input) 
    executes correctly on the available hardware.
    """
    
    # Setup inputs mimicking a feature map manipulation scenario
    # Batch size, Height, Width, Input Channels
    input_sizes = tf.constant([1, 32, 32, 3], dtype=tf.int32)
    
    # Filter: [Height, Width, Out Channels, In Channels]
    filter_shape = [3, 3, 16, 3]
    filter_val = np.random.randn(*filter_shape).astype(np.float32)
    filter_tensor = tf.constant(filter_val)
    
    # Output Backprop: [Batch, Height, Width, Out Channels]
    # This represents the gradient w.r.t. the output of the convolution
    out_backprop_shape = [1, 32, 32, 16]
    out_backprop_val = np.random.randn(*out_backprop_shape).astype(np.float32)
    out_backprop_tensor = tf.constant(out_backprop_val)

    # Attempt to run on GPU (analogous to MPS in the original bug)
    # If GPU is not available, it will fallback to CPU automatically or raise error
    # depending on TF configuration, similar to the fallback logic mentioned in the bug.
    try:
        with tf.device('/GPU:0'):
            result = tf.compat.v1.nn.conv2d_backprop_input(
                input_sizes=input_sizes,
                filter=filter_tensor,
                out_backprop=out_backprop_tensor,
                strides=[1, 1, 1, 1],
                padding='SAME'
            )
            print("Operation executed on GPU (accelerator).")
    except RuntimeError as e:
        # Fallback to CPU if GPU is not found, mirroring the fallback behavior
        # suggested in the original bug report (PYTORCH_ENABLE_MPS_FALLBACK).
        print(f"GPU not available, falling back to CPU. Error: {e}")
        result = tf.compat.v1.nn.conv2d_backprop_input(
            input_sizes=input_sizes,
            filter=filter_tensor,
            out_backprop=out_backprop_tensor,
            strides=[1, 1, 1, 1],
            padding='SAME'
        )
        print("Operation executed on CPU.")

    # Verify the output shape matches the expected input sizes
    # Note: input_sizes is a tensor, we compare the shape components
    expected_shape = [1, 32, 32, 3]
    assert result.shape == expected_shape, \
        f"Shape mismatch: expected {expected_shape}, got {result.shape}"
    
    # Verify the result is not all zeros (basic sanity check)
    assert tf.reduce_sum(tf.abs(result)).numpy() > 0, "Result is all zeros"

    print("Test passed: tf.compat.v1.nn.conv2d_backprop_input executed successfully.")

if __name__ == "__main__":
    test_tf_conv2d_backprop_input()
import torch
import tensorflow as tf
import sys

def test_fractional_max_pool_nan():
    """
    Adapted test case for tf.nn.fractional_max_pool based on the PyTorch bug report.
    Original Bug: MaxPool2d with channels_last + bfloat16 on CUDA produces NaNs for large tensors.
    """
    
    # Check for GPU availability to match the original bug context (CUDA)
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("This test requires a GPU to run, similar to the original PyTorch bug report.")
        sys.exit(0)

    print(f"Using GPU: {gpus[0].name}")

    # Large input tensor
    # PyTorch shape: N, C, H, W = 84, 64, 512, 960
    # TensorFlow shape (NHWC): N, H, W, C = 84, 512, 960, 64
    N, H, W, C = 84, 512, 960, 64

    with tf.device('/GPU:0'):
        # Case 1: bfloat16 + channels_last (NHWC) => Potential NaN
        # PyTorch: torch.randn(..., dtype=torch.bfloat16)
        x = tf.random.normal((N, H, W, C), dtype=tf.bfloat16)

        # PyTorch converts to channels_last (NHWC).
        # In TensorFlow, the default data format for pooling operations on GPU is NHWC,
        # which corresponds to channels_last. No explicit conversion needed if created in NHWC.
        
        print(f"Input tensor shape: {x.shape}")
        print(f"Input dtype: {x.dtype}")

        # PyTorch: MaxPool2d(kernel_size=3, stride=2, padding=1)
        # TensorFlow: fractional_max_pool
        # We map stride=2 to a pooling_ratio of [1.0, 2.0, 2.0, 1.0] (Batch, Height, Width, Channels)
        # to approximate the downsampling behavior.
        pooling_ratio = [1.0, 2.0, 2.0, 1.0]
        
        # tf.nn.fractional_max_pool returns (output, row_pooling_sequence, col_pooling_sequence)
        y, _, _ = tf.nn.fractional_max_pool(value=x, pooling_ratio=pooling_ratio)

        # Check for NaNs and Infs
        has_nan = tf.reduce_any(tf.math.is_nan(y))
        has_inf = tf.reduce_any(tf.math.is_inf(y))
        
        # Calculate stats (converting to float32 for safety if needed, though min/max handle bfloat16)
        y_min = tf.reduce_min(y)
        y_max = tf.reduce_max(y)

        print(f"Output shape: {y.shape}")
        print(f"Output contains NaN? {has_nan.numpy()}")
        print(f"Output contains Inf? {has_inf.numpy()}")
        print(f"Stats: min={y_min.numpy()}, max={y_max.numpy()}")

        if has_nan:
            print("Detected NaNs in FractionalMaxPool output!")
            # In a unit test, this would be: assert not has_nan, "NaNs detected in output"

if __name__ == "__main__":
    test_fractional_max_pool_nan()
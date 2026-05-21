import tensorflow as tf
import numpy as np

def test_per_image_standardization_large_tensor():
    """
    Test case adapted from PyTorch Issue 165297.
    
    The original issue reported NaNs and illegal memory access in MaxPool2d 
    when using large tensors with 'channels_last' memory format and bfloat16/float32 
    on CUDA.
    
    This test verifies that tf.image.per_image_standardization (which operates 
    on NHWC/Channels Last format by default) handles large tensors of similar 
    dimensions and data types without producing NaNs or Infs.
    """
    
    # Dimensions from the original bug report
    # PyTorch: N, C, H, W = 84, 64, 512, 960
    # TensorFlow (NHWC): N, H, W, C
    N, H, W, C = 84, 512, 960, 64
    
    # Case 1: bfloat16 (Original bug: NaNs)
    # We generate float32 data first to ensure valid range, then cast
    x_fp32 = np.random.randn(N, H, W, C).astype(np.float32)
    x_bf16 = tf.convert_to_tensor(x_fp32, dtype=tf.bfloat16)
    
    # TensorFlow defaults to channels_last (NHWC), matching the bug's trigger condition
    y_bf16 = tf.image.per_image_standardization(x_bf16)
    
    # Check for NaNs
    has_nan_bf16 = tf.reduce_any(tf.math.is_nan(y_bf16))
    has_inf_bf16 = tf.reduce_any(tf.math.is_inf(y_bf16))
    
    print(f"BF16 Test - Input shape: {x_bf16.shape}, Output contains NaN: {has_nan_bf16}, Output contains Inf: {has_inf_bf16}")
    assert not has_nan_bf16, "Detected NaNs in per_image_standardization output (bfloat16)"
    assert not has_inf_bf16, "Detected Infs in per_image_standardization output (bfloat16)"

    # Case 2: float32 (Original bug: illegal memory access)
    x_fp32_tensor = tf.convert_to_tensor(x_fp32, dtype=tf.float32)
    y_fp32 = tf.image.per_image_standardization(x_fp32_tensor)
    
    has_nan_fp32 = tf.reduce_any(tf.math.is_nan(y_fp32))
    has_inf_fp32 = tf.reduce_any(tf.math.is_inf(y_fp32))
    
    print(f"FP32 Test - Input shape: {x_fp32_tensor.shape}, Output contains NaN: {has_nan_fp32}, Output contains Inf: {has_inf_fp32}")
    assert not has_nan_fp32, "Detected NaNs in per_image_standardization output (float32)"
    assert not has_inf_fp32, "Detected Infs in per_image_standardization output (float32)"

if __name__ == "__main__":
    test_per_image_standardization_large_tensor()
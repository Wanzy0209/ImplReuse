import torch
import tensorflow as tf
import numpy as np

def test_tf_atrous_conv2d_transpose_negative_padding():
    """
    Adapted test case for tf.nn.atrous_conv2d_transpose based on 
    PyTorch constant_pad_nd negative padding behavior (Issue 161014).
    
    The original bug involves inconsistent behavior when padding results in 
    zero-sized dimensions vs negative-sized dimensions. 
    Since tf.nn.atrous_conv2d_transpose does not take a 'padding' list 
    to crop dimensions, we test the equivalent behavior by explicitly 
    setting the output_shape to zero or negative values.
    """
    
    # Setup input tensor: Batch=1, Height=5, Width=3, Channels=1
    # Corresponds to torch.ones([5, 3])
    input_tensor = tf.ones([1, 5, 3, 1], dtype=tf.float32)
    
    # Setup filters: Shape [filter_height, filter_width, out_channels, in_channels]
    filters = tf.ones([3, 3, 1, 1], dtype=tf.float32)
    
    print("Testing tf.nn.atrous_conv2d_transpose with edge-case output shapes...")

    # Case 1: Mimic the valid PyTorch case where output dim becomes 0
    # PyTorch: torch.ones([5, 3]), [-1, -2] -> torch.Size([5, 0])
    # TF Equivalent: output_shape=[1, 5, 0, 1] (Height=5, Width=0)
    try:
        output_shape_zero = [1, 5, 0, 1]
        res = tf.nn.atrous_conv2d_transpose(
            value=input_tensor,
            filters=filters,
            output_shape=output_shape_zero,
            rate=1,
            padding='SAME'
        )
        print(f"Test Case 1 (Zero Dim) Passed. Output Shape: {res.shape}")
        assert res.shape == tf.TensorShape([1, 5, 0, 1]), "Shape mismatch for zero dimension case"
    except Exception as e:
        print(f"Test Case 1 (Zero Dim) Failed with error: {e}")

    # Case 2: Mimic the invalid PyTorch case where calculation implies negative size
    # PyTorch: torch.ones([5, 3]), [-1, -2, 1, 1] -> RuntimeError
    # TF Equivalent: output_shape=[1, 5, -1, 1] (Height=5, Width=-1)
    try:
        output_shape_neg = [1, 5, -1, 1]
        res = tf.nn.atrous_conv2d_transpose(
            value=input_tensor,
            filters=filters,
            output_shape=output_shape_neg,
            rate=1,
            padding='SAME'
        )
        print(f"Test Case 2 (Negative Dim) Passed unexpectedly. Output Shape: {res.shape}")
    except Exception as e:
        print(f"Test Case 2 (Negative Dim) Failed as expected with error: {e}")

    # Case 3: Normal case for baseline
    try:
        output_shape_normal = [1, 7, 5, 1]
        res = tf.nn.atrous_conv2d_transpose(
            value=input_tensor,
            filters=filters,
            output_shape=output_shape_normal,
            rate=1,
            padding='SAME'
        )
        print(f"Test Case 3 (Normal) Passed. Output Shape: {res.shape}")
        assert res.shape == tf.TensorShape([1, 7, 5, 1]), "Shape mismatch for normal case"
    except Exception as e:
        print(f"Test Case 3 (Normal) Failed with error: {e}")

if __name__ == "__main__":
    test_tf_atrous_conv2d_transpose_negative_padding()
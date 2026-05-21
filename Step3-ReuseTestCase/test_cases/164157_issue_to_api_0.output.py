import torch
import tensorflow as tf
import numpy as np

def test_resize_volumes_fp16():
    """
    Test case for tf.keras.backend.resize_volumes based on the PyTorch issue 164157.
    
    The original issue involves an IncompatibleTypeErrorImpl related to fp16 and float64
    during eager vs compiled execution. This test verifies that resize_volumes handles
    float16 inputs correctly in both eager and graph (tf.function) modes to ensure
    no type divergence or errors occur.
    """
    
    # Setup: Create a 5D tensor with float16 dtype, mirroring the float16 usage in the bug report.
    # resize_volumes expects a 5D tensor. 
    # Using 'channels_last' format: (batch, depth, height, width, channels)
    x = tf.random.normal((2, 10, 10, 10, 3), dtype=tf.float16)
    
    depth_factor = 2
    height_factor = 2
    width_factor = 2
    data_format = 'channels_last'

    # 1. Test Eager Execution
    try:
        out_eager = tf.keras.backend.resize_volumes(
            x, depth_factor, height_factor, width_factor, data_format
        )
        print("Eager Success! ")
    except Exception as e:
        print(f"Eager Failed: {e}")
        return

    # 2. Test Compiled Execution (tf.function) - analogous to torch.compile
    @tf.function
    def compiled_resize_volumes(x):
        return tf.keras.backend.resize_volumes(
            x, depth_factor, height_factor, width_factor, data_format
        )

    try:
        out_compiled = compiled_resize_volumes(x)
        print("Compile Success! ")
    except Exception as e:
        print(f"Compile Failed: {e}")
        return

    # 3. Verify Consistency
    # Ensure the compiled output matches the eager output (no divergence)
    if tf.reduce_all(tf.equal(out_eager, out_compiled)).numpy():
        print("Consistency Check Passed! ")
    else:
        print("Consistency Check Failed! ")

if __name__ == '__main__':
    test_resize_volumes_fp16()
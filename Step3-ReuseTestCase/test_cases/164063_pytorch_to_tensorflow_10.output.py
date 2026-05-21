import torch
import tensorflow as tf
import numpy as np

def test_arctan2_bfloat16():
    """
    Test case for tf.keras.ops.arctan2 adapted from a PyTorch bug report 
    involving torch.var with bfloat16 inputs.
    
    The original bug involved a TypeError related to fp32 types when processing
    bfloat16 tensors in a compiled context. This test verifies that 
    tf.keras.ops.arctan2 handles bfloat16 inputs correctly in both eager 
    and compiled (tf.function) modes.
    """
    
    # Mimic the bfloat16 context from the PyTorch bug
    # PyTorch t1 shape: (28, 24, 3, 127)
    shape = (28, 24, 3, 127)
    dtype = tf.bfloat16

    # Create bfloat16 inputs
    # Using random values to ensure dynamic execution paths
    x1 = tf.random.uniform(shape, minval=-10.0, maxval=10.0, dtype=dtype)
    x2 = tf.random.uniform(shape, minval=0.1, maxval=10.0, dtype=dtype) # Avoid division by zero

    # 1. Test Eager Execution
    try:
        out_eager = tf.keras.ops.arctan2(x1, x2)
        # Verify output is valid
        assert out_eager is not None
        assert not tf.reduce_any(tf.math.is_nan(out_eager)), "NaN found in eager output"
        print(f'Eager Success!  Output dtype: {out_eager.dtype}')
    except Exception as e:
        print(f'Eager Failed!  Error: {e}')
        return

    # 2. Test Compiled Execution (tf.function)
    # This mimics the torch.compile behavior in the original bug report
    @tf.function(jit_compile=True)
    def compiled_arctan2(a, b):
        return tf.keras.ops.arctan2(a, b)

    try:
        out_compiled = compiled_arctan2(x1, x2)
        # Verify output is valid
        assert out_compiled is not None
        assert not tf.reduce_any(tf.math.is_nan(out_compiled)), "NaN found in compiled output"
        print(f'Compile Success!  Output dtype: {out_compiled.dtype}')
    except Exception as e:
        print(f'Compile Failed!  Error: {e}')

if __name__ == '__main__':
    test_arctan2_bfloat16()
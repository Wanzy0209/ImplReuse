import torch
import tensorflow as tf
import numpy as np

def test_argmax_fp16_compile_divergence():
    """
    Test case adapted from Issue 164157.
    Original Issue: IncompatibleTypeErrorImpl in torch.std with float16 inputs during compilation.
    Similar API: tf.math.argmax.
    
    This test verifies that tf.math.argmax behaves consistently between eager and compiled (XLA)
    execution when operating on float16 tensors, mirroring the context of the original bug.
    """

    # Define the computation logic mimicking the original bug's structure
    # Original: t9 (float16) -> std(dim=2) -> output
    # Adapted: t9 (float16) -> argmax(axis=2) -> output
    def foo(arg3, arg4, arg5):
        # Inputs are float16 tensors, matching the original bug's dtype context
        t6 = arg3
        t7 = arg4
        t8 = arg5
        
        # Concatenate to create the specific tensor shape used in the bug report
        # t9 shape: (256, 88, 4), dtype: float16
        t9 = tf.concat([t6, t6, t7, t8], axis=2)
        
        # Apply the Similar API: tf.math.argmax
        # Original API was torch.std. Both are reduction operations along a specific dimension.
        t10 = tf.math.argmax(t9, axis=2)
        
        return t10

    # Initialize inputs with float16 dtype, matching the original bug report
    # Shapes: (256, 88, 1)
    arg3 = tf.random.normal([256, 88, 1], dtype=tf.float16)
    arg4 = tf.random.normal([256, 88, 1], dtype=tf.float16)
    arg5 = tf.random.normal([256, 88, 1], dtype=tf.float16)

    # 1. Run in Eager mode
    out_eager = foo(arg3, arg4, arg5)
    print('Eager Execution Success! ')

    # 2. Run in Compiled mode (using tf.function with JIT compilation to mimic torch.compile)
    # The original bug manifested specifically during the compilation phase (Triton).
    compiled_foo = tf.function(foo, jit_compile=True)
    
    try:
        out_compiled = compiled_foo(arg3, arg4, arg5)
        print('Compile Execution Success! ')
    except Exception as e:
        print(f'Compile Execution Failed! : {e}')
        raise

    # 3. Verify consistency (Divergence check)
    # The original bug was a crash, but checking output equality ensures semantic consistency.
    # Note: argmax returns int64, so we compare indices.
    assert tf.reduce_all(tf.equal(out_eager, out_compiled)).numpy(), \
        "Divergence detected between eager and compiled outputs!"
    
    print("Test Passed: No divergence between eager and compiled execution.")

if __name__ == '__main__':
    test_argmax_fp16_compile_divergence()
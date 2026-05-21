import torch
import tensorflow as tf
import tf.experimental.numpy as tnp

# Enable numpy behavior for tf.experimental.numpy
tf.experimental.numpy.experimental_enable_numpy_behavior()

def foo(arg0, arg1):
    # The original bug report highlights a type mismatch issue (fp16 vs float64) 
    # during compilation with torch.std.
    # We adapt this to test tf.experimental.numpy.gcd, which involves 
    # _promote_dtype_binary logic according to the similar API information.
    # We use float16 inputs to stress the type promotion and compilation path.
    return tnp.gcd(arg0, arg1)

# Inputs adapted from the original bug report (arg3, arg4)
# Original: size=(256, 88, 1), dtype=float16, device=cuda
# We use float16 to test type handling similar to the original bug.
arg0 = tf.random.uniform([256, 88, 1], minval=0, maxval=100, dtype=tf.float16)
arg1 = tf.random.uniform([256, 88, 1], minval=0, maxval=100, dtype=tf.float16)

if __name__ == '__main__':
    # 1. Eager Execution
    try:
        out_eager = foo(arg0, arg1)
        print('Eager Success! ')
    except Exception as e:
        print(f'Eager Failed: {e}')
        raise

    # 2. Compiled Execution
    # tf.function with jit_compile=True is analogous to torch.compile (using XLA/Triton backends)
    try:
        compiled_foo = tf.function(foo, jit_compile=True)
        out_compiled = compiled_foo(arg0, arg1)
        print('Compile Success! ')
    except Exception as e:
        print(f'Compile Failed: {e}')
        raise

    # 3. Consistency Check
    # Verify that the compiled output matches the eager output
    if tf.reduce_all(tf.equal(out_eager, out_compiled)).numpy():
        print('Consistency Check Passed! ')
    else:
        print('Consistency Check Failed! ')
        raise AssertionError("Eager and Compiled outputs differ")
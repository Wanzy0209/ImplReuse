import torch
import tensorflow as tf
import numpy as np

def foo(x1, x2):
    """
    Adapted function using tf.experimental.numpy.arctan2.
    The original bug involved torch.var on bfloat16 tensors.
    We test arctan2 with bfloat16 tensors to check for similar type handling issues.
    """
    return tf.experimental.numpy.arctan2(x1, x2)

# Replicating the shape and dtype from the original bug report (t1)
# size=(28, 24, 3, 127), dtype=bfloat16
shape = (28, 24, 3, 127)
dtype = tf.bfloat16

# Create inputs
# Note: We use random values to ensure the operation is actually executed
x1 = tf.random.normal(shape, dtype=dtype)
x2 = tf.random.normal(shape, dtype=dtype)

if __name__ == '__main__':
    # 1. Eager Execution
    try:
        out_eager = foo(x1, x2)
        # Basic assertion to ensure output shape matches input shape (element-wise op)
        assert out_eager.shape == shape, f"Shape mismatch: expected {shape}, got {out_eager.shape}"
        print('Eager Success! ')
    except Exception as e:
        print(f'Eager Failed: {e}')

    # 2. Compiled Execution (tf.function with XLA)
    # This mimics the torch.compile(fullgraph=True) behavior in the original bug report
    try:
        compiled_foo = tf.function(foo, jit_compile=True)
        out_compiled = compiled_foo(x1, x2)
        assert out_compiled.shape == shape, f"Shape mismatch: expected {shape}, got {out_compiled.shape}"
        print('Compile Success! ')
    except Exception as e:
        print(f'Compile Failed: {e}')
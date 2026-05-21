import torch
import tensorflow as tf
import numpy as np

def foo(a, b):
    # The original bug involves torch.var on bfloat16 tensors.
    # We adapt this to test tf.experimental.numpy.outer with bfloat16 inputs.
    # The implementation of outer involves reshaping and multiplication,
    # which triggers type promotion logic similar to the variance calculation.
    return tf.experimental.numpy.outer(a, b)

# Setup inputs with bfloat16 dtype, matching the context of the original bug
# Original: arg0 = torch.rand(..., dtype=torch.bfloat16, ...)
a = tf.constant([1.0, 2.0, 3.0], dtype=tf.bfloat16)
b = tf.constant([4.0, 5.0, 6.0], dtype=tf.bfloat16)

if __name__ == '__main__':
    # Test Eager Execution
    try:
        out_eager = foo(a, b)
        print(f"Eager Result: {out_eager.numpy()}")
        print('Eager Success! ')
    except Exception as e:
        print(f'Eager Failed: {e}')

    # Test Compiled Execution (tf.function)
    # This mimics the torch.compile behavior in the original bug report
    try:
        compiled_foo = tf.function(foo)
        out_compiled = compiled_foo(a, b)
        print(f"Compiled Result: {out_compiled.numpy()}")
        print('Compile Success! ')
    except Exception as e:
        print(f'Compile Failed: {e}')
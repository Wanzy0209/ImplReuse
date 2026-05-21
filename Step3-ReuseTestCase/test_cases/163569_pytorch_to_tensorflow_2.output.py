import torch
import tensorflow as tf
import numpy as np

def foo(arg0, arg1, arg2):
    # t0 = arg0 # size=(2, 261, 17, 358), dtype=bfloat16
    # t1 = t0.max(dim=0).values
    t1 = tf.reduce_max(arg0, axis=0) # size=(261, 17, 358)
    # t2 = t1.transpose(1, 0)
    t2 = tf.transpose(t1, perm=[1, 0, 2]) # size=(17, 261, 358)

    # t3 = arg1 # size=(17, 64, 358), dtype=float32
    # t4 = torch.exp(t3)
    t4 = tf.exp(arg1) # size=(17, 64, 358)

    # Original: t7 = torch.nn.functional.conv1d(t4, t6, stride=1, padding=0)
    # Adaptation: Use tf.keras.ops.prod.
    # To maintain shape compatibility for the subsequent multiplication (t2 * t7 * t8),
    # we need t7 to broadcast with t2 (17, 261, 358).
    # t4 is (17, 64, 358). Reducing axis 1 with keepdims=True gives (17, 1, 358).
    # This broadcasts correctly with t2.
    t7 = tf.keras.ops.prod(t4, axis=1, keepdims=True)

    # t8 = t7.clone(); t8.zero_()
    t8 = tf.identity(t7)
    t8 = tf.zeros_like(t8)

    # t9 = t2 * t7 * t8
    t9 = t2 * t7 * t8
    output = t9
    return output

# Setup inputs
# Note: PyTorch uses 'cuda', TF uses GPU if available. We'll use standard TF logic.
arg0 = tf.random.normal([2, 261, 17, 358], dtype=tf.bfloat16)
arg1 = tf.random.normal([17, 64, 358], dtype=tf.float32)
arg2 = tf.random.normal([261, 1, 64], dtype=tf.float32)

if __name__ == '__main__':
    # Eager execution
    out_eager = foo(arg0, arg1, arg2)
    print('Eager Success! ')

    # Compiled execution (tf.function with XLA)
    # Using jit_compile=True to stress the compiler like torch.compile
    compiled_foo = tf.function(foo, jit_compile=True)
    try:
        out_compiled = compiled_foo(arg0, arg1, arg2)
        print('Compile Success! ')

        # Check for divergence
        if not np.allclose(out_eager.numpy(), out_compiled.numpy(), atol=1e-5):
            print("Divergence detected between eager and compiled outputs!")
        else:
            print("Outputs match.")
    except Exception as e:
        print(f"Compile Failed! : {e}")
import torch
import numpy as np
import sys

# Attempt to import TensorFlow, handle environment errors gracefully
try:
    import tensorflow as tf
except ImportError as e:
    print("Skipping test: Failed to import TensorFlow.")
    print("This is likely due to a system environment issue (e.g., GLIBC version mismatch).")
    print(f"Details: {e}")
    sys.exit(0)

# Enable mixed precision to mimic the fp16 context of the original bug
# The original bug involved 'invalid operands of type pointer<fp16> and triton.language.float64'
tf.keras.mixed_precision.set_global_policy('mixed_float16')

def foo(arg0, arg1, arg2, arg3, arg4, arg5, sentinel):
    # t0 = arg0 # size=(42, 56), dtype=int64
    # PyTorch tanh on int64 works, TF requires float. Cast to float16 to match mixed precision context.
    t0_float = tf.cast(arg0, tf.float16)
    t1 = tf.tanh(t0_float) # size=(42, 56), dtype=float16

    # t2 = t1.clone(); t2.zero_()
    t2 = tf.zeros_like(t1)

    # t3 = arg1 # size=(50000, 128), dtype=float16
    # t4 = arg2 # size=(46, 128), dtype=float16
    # t5 = torch.nn.functional.linear(t3, t4) -> matmul(t3, t4.T)
    t5 = tf.linalg.matmul(t3, t4, transpose_b=True) # size=(50000, 46)

    # t6 = arg3 # size=(50000, 4, 46), dtype=float16
    # Original: t7 = t6.max(dim=1).values
    # --- SIMILAR API USAGE ---
    # Using tf.experimental.numpy.argmax instead of torch.max.
    # Note: argmax returns indices (int64), so we cast to float16 to maintain the graph flow
    # similar to the original bug where t7 (float16) was used in pow.
    t7_indices = tf.experimental.numpy.argmax(t6, axis=1)
    t7 = tf.cast(t7_indices, tf.float16)

    # t8 = arg4 # size=(25786, 46), dtype=float16
    # t9 = arg5 # size=(24214, 46), dtype=float16
    # t10 = torch.cat([t8, t9], dim=0)
    t10 = tf.concat([t8, t9], axis=0)

    # t11 = torch.pow(torch.pow(torch.pow(torch.pow(t5, t7), t10), t5), t7)
    # This chain of operations stresses the type system, similar to the original bug.
    t11 = tf.pow(tf.pow(tf.pow(tf.pow(t5, t7), t10), t5), t7)

    # t12 = torch.nn.functional.embedding(...)
    # Clamp indices
    max_idx = tf.cast(tf.shape(t11)[0] - 1, tf.int32)
    indices = tf.clip_by_value(tf.cast(t2, tf.int32), 0, max_idx)
    t12 = tf.nn.embedding_lookup(t11, indices)

    output = t12 + sentinel
    return output

# Generate inputs matching the original bug report shapes and types
arg0 = tf.random.uniform((42, 56), minval=0, maxval=1000, dtype=tf.int64)
arg1 = tf.random.uniform((50000, 128), dtype=tf.float16)
arg2 = tf.random.uniform((46, 128), dtype=tf.float16)
arg3 = tf.random.uniform((50000, 4, 46), dtype=tf.float16)
arg4 = tf.random.uniform((25786, 46), dtype=tf.float16)
arg5 = tf.random.uniform((24214, 46), dtype=tf.float16)
sentinel = tf.constant(0.0, dtype=tf.float16)

if __name__ == '__main__':
    print("Running Eager Mode...")
    try:
        out_eager = foo(arg0, arg1, arg2, arg3, arg4, arg5, sentinel)
        print('Eager Success! ')
    except Exception as e:
        print(f'Eager Failed! : {e}')

    print("\nRunning Compiled Mode (tf.function)...")
    try:
        compiled_foo = tf.function(foo)
        out_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4, arg5, sentinel)
        print('Compile Success! ')
        
        # Check for divergence between eager and compiled
        if tf.reduce_all(tf.abs(out_eager - out_compiled) < 1e-2):
            print("Outputs match. ")
        else:
            print("Outputs diverge! ")
    except Exception as e:
        print(f'Compile Failed! : {e}')
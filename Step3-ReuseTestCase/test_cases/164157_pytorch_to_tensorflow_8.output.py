import torch
import tensorflow as tf
import numpy as np

def foo(arg0, arg1, arg2, arg3, arg4, arg5, sentinel):
    # t0 = arg0 # size=(47,), stride=(1,), dtype=int64
    t0 = arg0

    # t1 = torch.tanh(t0) # size=(47,), stride=(1,), dtype=int64
    # Note: PyTorch tanh on int64 implicitly casts to float. TF requires explicit cast.
    t1 = tf.math.tanh(tf.cast(t0, tf.float32))

    # t2 = arg1 # size=(), stride=(), dtype=int64
    t2 = arg1
    # t3 = arg2 # size=(), stride=(), dtype=int64
    t3 = arg2
    # t4 = t2 * t3
    t4 = t2 * t3

    # t5 = t1.clone(); t5.fill_(t4.item())
    # TF equivalent: create a tensor of t1's shape filled with t4's value
    t5 = tf.fill(tf.shape(t1), tf.cast(t4, tf.float32))

    # t6 = arg3 # size=(256, 88, 1), dtype=float16
    t6 = arg3
    # t7 = arg4 # size=(256, 88, 1), dtype=float16
    t7 = arg4
    # t8 = arg5 # size=(256, 88, 1), dtype=float16
    t8 = arg5

    # t9 = torch.cat([t6, t6, t7, t8], dim=2) # size=(256, 88, 4), dtype=float16
    t9 = tf.concat([t6, t6, t7, t8], axis=2)

    # --- ADAPTED SECTION ---
    # Original: t10 = t9.std(dim=2) # size=(256, 88), dtype=float16
    # Target API: tf.keras.ops.power
    # We apply power to the float16 tensor. To maintain the shape compatibility 
    # for the subsequent embedding lookup (which expects (256, 88)), we perform 
    # a reduction (mean) on the result of the power operation.
    t_power = tf.keras.ops.power(t9, tf.constant(2.0, dtype=tf.float16))
    t10 = tf.reduce_mean(t_power, axis=2) # Reduces to (256, 88)
    # ----------------------

    # t11 = torch.nn.functional.embedding(torch.clamp(t5, 0, t10.size(0) - 1).to(torch.long), t10)
    # Clamp indices to valid range for embedding
    indices = tf.clip_by_value(tf.cast(t5, tf.int32), 0, tf.shape(t10)[0] - 1)
    # Embedding lookup
    t11 = tf.nn.embedding_lookup(t10, indices)

    # output = t11 + sentinel
    output = t11 + sentinel
    return output

# Setup inputs
# arg0 = torch.randint(0, 1000, [47], dtype=torch.int64, device='cuda')
arg0 = tf.random.uniform((47,), minval=0, maxval=1000, dtype=tf.int64)

# arg1 = torch.randint(0, 1000, [], dtype=torch.int64, device='cuda')
arg1 = tf.random.uniform((), minval=0, maxval=1000, dtype=tf.int64)

# arg2 = torch.randint(0, 1000, [], dtype=torch.int64, device='cuda')
arg2 = tf.random.uniform((), minval=0, maxval=1000, dtype=tf.int64)

# arg3 = torch.rand([256, 88, 1], dtype=torch.float16, device='cuda', requires_grad=True)
arg3 = tf.random.uniform((256, 88, 1), dtype=tf.float16)
arg4 = tf.random.uniform((256, 88, 1), dtype=tf.float16)
arg5 = tf.random.uniform((256, 88, 1), dtype=tf.float16)

# sentinel = torch.tensor(0.0, dtype=torch.float16, device='cuda', requires_grad=True)
sentinel = tf.Variable(0.0, dtype=tf.float16)

if __name__ == '__main__':
    # Eager Execution
    out_eager = foo(arg0, arg1, arg2, arg3, arg4, arg5, sentinel)
    print('Eager Success! ')

    # Compiled Execution (XLA)
    # Using jit_compile=True to mimic torch.compile behavior
    compiled_foo = tf.function(foo, jit_compile=True)
    
    try:
        out_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4, arg5, sentinel)
        print('Compile Success! ')

        # Check for divergence (Eager vs Compile)
        # Note: float16 operations might have slight precision differences, 
        # but we check for significant divergence or crashes.
        if not np.allclose(out_eager.numpy(), out_compiled.numpy(), rtol=1e-2, atol=1e-2):
            print("Divergence detected between Eager and Compiled modes!")
        else:
            print("Results match between Eager and Compiled modes.")
    except Exception as e:
        print(f"Compile Failed with error: {e}")
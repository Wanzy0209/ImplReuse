import torch
import tensorflow as tf
import numpy as np

def foo(arg0, arg1, arg2, arg3, arg4, arg5, sentinel):
    # t0 = arg0
    t0 = arg0
    # t1 = torch.tanh(t0) -> tf.math.tanh requires float cast
    t1 = tf.math.tanh(tf.cast(t0, tf.float32))
    t1 = tf.cast(t1, tf.int64)

    # t2 = arg1, t3 = arg2
    t2 = arg1
    t3 = arg2
    # t4 = t2 * t3
    t4 = t2 * t3
    # t5 = t1.clone(); t5.fill_(t4.item())
    # TF equivalent: create a tensor filled with the value of t4
    t5 = tf.fill(tf.shape(t1), tf.cast(t4, tf.int64))

    # t6, t7, t8
    t6 = arg3
    t7 = arg4
    t8 = arg5
    # t9 = torch.cat([t6, t6, t7, t8], dim=2)
    t9 = tf.concat([t6, t6, t7, t8], axis=2)

    # --- SIMILAR API USAGE ---
    # Original: t10 = t9.std(dim=2)
    # Similar: t10 = tf.keras.ops.argmax(t9, axis=2)
    # This tests the reduction operation on float16 tensors.
    t10 = tf.keras.ops.argmax(t9, axis=2)

    # t11 = torch.nn.functional.embedding(torch.clamp(t5, 0, t10.size(0) - 1).to(torch.long), t10)
    # TF equivalent: tf.nn.embedding_lookup(params, ids)
    # Here t10 (result of argmax) is used as params (weights), and t5 as ids.
    # Note: t10 is int64 due to argmax, whereas original was float16.
    ids = tf.clip_by_value(t5, 0, tf.shape(t10)[0] - 1)
    ids = tf.cast(ids, tf.int32)
    t11 = tf.nn.embedding_lookup(t10, ids)

    # output = t11 + sentinel
    output = t11 + sentinel
    return output

# Input setup mirroring the original issue
# arg0: size=(47,), dtype=int64
arg0 = tf.constant(np.random.randint(0, 1000, [47]), dtype=tf.int64)
# arg1, arg2: size=(), dtype=int64
arg1 = tf.constant(np.random.randint(0, 1000, []), dtype=tf.int64)
arg2 = tf.constant(np.random.randint(0, 1000, []), dtype=tf.int64)
# arg3, arg4, arg5: size=(256, 88, 1), dtype=float16
arg3 = tf.random.uniform([256, 88, 1], dtype=tf.float16)
arg4 = tf.random.uniform([256, 88, 1], dtype=tf.float16)
arg5 = tf.random.uniform([256, 88, 1], dtype=tf.float16)
# sentinel: float16
sentinel = tf.constant(0.0, dtype=tf.float16)

if __name__ == '__main__':
    # Eager execution
    out_eager = foo(arg0, arg1, arg2, arg3, arg4, arg5, sentinel)
    print('Eager Success! ')

    # Compiled execution (mimicking torch.compile with fullgraph=True)
    # Using jit_compile=True to strictly check for compilation errors similar to the Triton backend
    compiled_foo = tf.function(foo, jit_compile=True)
    out_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4, arg5, sentinel)
    print('Compile Success! ')

    # Check for divergence
    if tf.reduce_all(tf.equal(out_eager, out_compiled)):
        print('Results Match! ')
    else:
        print('Results Diverge! ')
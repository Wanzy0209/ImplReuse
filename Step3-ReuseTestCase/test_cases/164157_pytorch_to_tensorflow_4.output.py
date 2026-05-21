import torch
import tensorflow as tf
import numpy as np

def foo(arg0, arg1, arg2, arg3, arg4, arg5, sentinel):
    # t0 = arg0 # size=(47,), dtype=int64
    # t1 = torch.tanh(t0) # PyTorch allows tanh on int, TF requires float
    t1 = tf.cast(tf.tanh(tf.cast(arg0, tf.float32)), tf.int64)
    
    # t2, t3, t4 setup
    t2 = arg1
    t3 = arg2
    t4 = t2 * t3
    
    # t5 = t1.clone(); t5.fill_(t4.item())
    t5 = tf.fill(tf.shape(t1), tf.cast(t4, tf.int64))
    
    # t6, t7, t8 setup (float16)
    t6 = arg3
    t7 = arg4
    t8 = arg5
    
    # t9 = torch.cat([t6, t6, t7, t8], dim=2)
    t9 = tf.concat([t6, t6, t7, t8], axis=2)
    
    # t10 = t9.std(dim=2) -> Replaced with tf.math.nextafter
    # Original bug context: float16 operations leading to type errors in compilation.
    # tf.math.nextafter is element-wise, so shape remains (256, 88, 4).
    # We ensure inputs are float16 to test type stability.
    # Using t9 and a slightly modified t9 as inputs to nextafter.
    t10 = tf.math.nextafter(t9, tf.cast(t9 + 0.1, tf.float16))
    
    # t11 = torch.nn.functional.embedding(...)
    # Adaptation: tf.nn.embedding_lookup
    # Clamp indices
    # t10.size(0) is 256
    max_idx = tf.shape(t10)[0] - 1
    clamped_ids = tf.clip_by_value(t5, 0, max_idx)
    # Cast to int32 for embedding_lookup (standard requirement)
    clamped_ids = tf.cast(clamped_ids, tf.int32)
    
    # embedding lookup using t10 as weights
    t11 = tf.nn.embedding_lookup(t10, clamped_ids)
    
    output = t11 + sentinel
    return output

# Input generation
arg0 = tf.constant(np.random.randint(0, 1000, [47]), dtype=tf.int64)
arg1 = tf.constant(np.random.randint(0, 1000, []), dtype=tf.int64)
arg2 = tf.constant(np.random.randint(0, 1000, []), dtype=tf.int64)
arg3 = tf.random.normal([256, 88, 1], dtype=tf.float16)
arg4 = tf.random.normal([256, 88, 1], dtype=tf.float16)
arg5 = tf.random.normal([256, 88, 1], dtype=tf.float16)
sentinel = tf.constant(0.0, dtype=tf.float16)

if __name__ == '__main__':
    # Eager Execution
    out_eager = foo(arg0, arg1, arg2, arg3, arg4, arg5, sentinel)
    print('Eager Success! ')

    # Compiled Execution (tf.function with XLA)
    # This mimics the torch.compile behavior where the bug was found
    compiled_foo = tf.function(foo, jit_compile=True)
    out_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4, arg5, sentinel)
    print('Compile Success! ')

    # Verify consistency
    # Check shapes match
    assert out_eager.shape == out_compiled.shape, "Shape divergence detected"
    
    # Check values are close (accounting for potential float16 precision differences in different execution paths)
    diff = tf.reduce_max(tf.abs(tf.cast(out_eager, tf.float32) - tf.cast(out_compiled, tf.float32)))
    assert diff < 1e-2, f"Value divergence detected: {diff}"
    
    print("Test Passed: No divergence between Eager and Compiled modes.")
import torch
import tensorflow as tf
import tf.experimental.numpy as tnp

def foo(arg0, arg1, arg2, arg3, arg4, arg5, sentinel):
    # Translate int64 operations
    # PyTorch tanh on int64 returns int64, TF requires float inputs for tanh
    t0 = arg0
    t1 = tf.cast(tf.tanh(tf.cast(t0, tf.float32)), tf.int64)
    
    t2 = arg1
    t3 = arg2
    t4 = t2 * t3
    
    # t5 = t1.clone(); t5.fill_(t4.item())
    # Extract scalar value and fill tensor
    t5_val = tf.cast(t4, tf.int64)
    t5 = tf.fill(tf.shape(t1), t5_val)
    
    # Translate float16 operations
    t6 = arg3
    t7 = arg4
    t8 = arg5
    t9 = tf.concat([t6, t6, t7, t8], axis=2) # shape (256, 88, 4)
    
    # --- API Adaptation ---
    # Original: t10 = t9.std(dim=2)
    # Similar API: tf.experimental.numpy.nextafter
    # We apply nextafter to t9. To maintain the flow for the subsequent embedding operation,
    # we reduce the result back to the expected shape (256, 88).
    # The original bug involved fp16 and float64 incompatibility. 
    # t9 is float16. t9 + 1.0 promotes to float32. nextafter handles precision boundaries.
    t10_next = tnp.nextafter(t9, t9 + 1.0)
    t10 = tf.reduce_mean(t10_next, axis=2) # Reduce to (256, 88) to match embedding weights shape
    
    # Embedding operation
    # t11 = torch.nn.functional.embedding(torch.clamp(t5, 0, t10.size(0) - 1).to(torch.long), t10)
    indices = tf.clip_by_value(t5, 0, tf.shape(t10)[0] - 1)
    indices = tf.cast(indices, tf.int32) # TF embedding_lookup expects int32
    t11 = tf.nn.embedding_lookup(t10, indices)
    
    output = t11 + sentinel
    return output

# Generate inputs
arg0 = tf.random.uniform([47], minval=0, maxval=1000, dtype=tf.int64)
arg1 = tf.random.uniform([], minval=0, maxval=1000, dtype=tf.int64)
arg2 = tf.random.uniform([], minval=0, maxval=1000, dtype=tf.int64)
arg3 = tf.random.uniform([256, 88, 1], dtype=tf.float16)
arg4 = tf.random.uniform([256, 88, 1], dtype=tf.float16)
arg5 = tf.random.uniform([256, 88, 1], dtype=tf.float16)
sentinel = tf.constant(0.0, dtype=tf.float16)

if __name__ == '__main__':
    # Test Eager Execution
    print("Testing Eager Execution...")
    try:
        out_eager = foo(arg0, arg1, arg2, arg3, arg4, arg5, sentinel)
        # Check output properties
        assert out_eager.dtype == tf.float16, "Output dtype mismatch"
        assert out_eager.shape == (47, 88), "Output shape mismatch"
        print('Eager Success! ')
    except Exception as e:
        print(f'Eager Failed: {e}')

    # Test Compiled Execution (tf.function)
    print("\nTesting Compiled Execution...")
    try:
        compiled_foo = tf.function(foo)
        out_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4, arg5, sentinel)
        assert out_compiled.dtype == tf.float16, "Compiled output dtype mismatch"
        assert out_compiled.shape == (47, 88), "Compiled output shape mismatch"
        print('Compile Success! ')
    except Exception as e:
        print(f'Compile Failed: {e}')
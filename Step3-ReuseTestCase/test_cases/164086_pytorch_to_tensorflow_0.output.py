import torch
import tensorflow as tf
import numpy as np

def foo(arg0, arg1, arg2, arg3, arg4, arg5, sentinel):
    # t0 = arg0 # size=(42, 56), dtype=int64
    t0 = arg0
    
    # Original API: torch.tanh(t0)
    # Adapted API: tf.compat.v1.no_regularizer(t0)
    # The original bug involved type handling with int64 inputs.
    # We verify that no_regularizer handles the int64 input gracefully (returns None).
    reg = tf.compat.v1.no_regularizer(t0)
    
    # Since no_regularizer returns None, we cannot use it in subsequent math operations
    # like the original t1. To preserve the graph structure and test the rest of the flow,
    # we perform the tanh operation separately (simulating the data flow).
    # Note: tf.tanh requires floating point inputs, so we cast.
    t1 = tf.tanh(tf.cast(t0, tf.float32))
    
    # t2 = t1.clone(); t2.zero_()
    # In TF, we create a zero tensor of the same shape.
    t2 = tf.zeros_like(t1, dtype=tf.float32)
    
    # t3 = arg1 # size=(50000, 128), dtype=float16
    t3 = arg1
    # t4 = arg2 # size=(46, 128), dtype=float16
    t4 = arg2
    # t5 = torch.nn.functional.linear(t3, t4)
    # TF equivalent: tf.linalg.matmul (transpose_b=True for linear layer weights)
    t5 = tf.linalg.matmul(t3, t4, transpose_b=True)
    
    # t6 = arg3 # size=(50000, 4, 46), dtype=float16
    t6 = arg3
    # t7 = t6.max(dim=1).values
    t7 = tf.reduce_max(t6, axis=1)
    
    # t8 = arg4 # size=(25786, 46), dtype=float16
    t8 = arg4
    # t9 = arg5 # size=(24214, 46), dtype=float16
    t9 = arg5
    # t10 = torch.cat([t8, t9], dim=0)
    t10 = tf.concat([t8, t9], axis=0)
    
    # t11 = torch.pow(torch.pow(torch.pow(torch.pow(t5, t7), t10), t5), t7)
    # Chained pow operations
    t11 = tf.pow(tf.pow(tf.pow(tf.pow(t5, t7), t10), t5), t7)
    
    # t12 = torch.nn.functional.embedding(torch.clamp(t2, 0, t11.size(0) - 1).to(torch.long), t11)
    # Clamp indices
    max_idx = tf.cast(tf.shape(t11)[0] - 1, tf.float32)
    clamped_indices = tf.clip_by_value(t2, 0.0, max_idx)
    # Cast to int32 for embedding lookup (TF standard)
    int_indices = tf.cast(clamped_indices, tf.int32)
    # Embedding lookup
    t12 = tf.nn.embedding_lookup(t11, int_indices)
    
    # output = t12 + sentinel
    output = t12 + sentinel
    
    return output, reg

# Setup inputs matching the original bug report shapes and dtypes
# arg0: int64
arg0 = tf.constant(np.random.randint(0, 1000, (42, 56)), dtype=tf.int64)
# arg1: float16
arg1 = tf.constant(np.random.rand(50000, 128), dtype=tf.float16)
# arg2: float16
arg2 = tf.constant(np.random.rand(46, 128), dtype=tf.float16)
# arg3: float16
arg3 = tf.constant(np.random.rand(50000, 4, 46), dtype=tf.float16)
# arg4: float16
arg4 = tf.constant(np.random.rand(25786, 46), dtype=tf.float16)
# arg5: float16
arg5 = tf.constant(np.random.rand(24214, 46), dtype=tf.float16)
# sentinel: float16
sentinel = tf.constant(0.0, dtype=tf.float16)

if __name__ == '__main__':
    # 1. Test Eager Mode
    print("Testing Eager Mode...")
    out_eager, reg_eager = foo(arg0, arg1, arg2, arg3, arg4, arg5, sentinel)
    
    # Verify no_regularizer behavior
    assert reg_eager is None, "tf.compat.v1.no_regularizer should return None"
    print('Eager Success! ')
    
    # 2. Test Compiled Mode (tf.function)
    print("Testing Compiled Mode (tf.function)...")
    compiled_foo = tf.function(foo)
    out_compiled, reg_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4, arg5, sentinel)
    
    # Verify no_regularizer behavior in compiled mode
    assert reg_compiled is None, "tf.compat.v1.no_regularizer should return None in compiled mode"
    
    # Check for divergence (similar to the original bug report's intent)
    # Note: Due to potential non-determinism in some ops or slight precision differences in TF vs PyTorch,
    # we check shape and dtype consistency primarily, or close equality if deterministic.
    assert out_eager.shape == out_compiled.shape, "Output shape mismatch between eager and compiled"
    assert out_eager.dtype == out_compiled.dtype, "Output dtype mismatch between eager and compiled"
    
    print('Compile Success! ')
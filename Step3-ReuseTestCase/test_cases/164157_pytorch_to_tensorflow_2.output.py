import torch
import tensorflow as tf
import numpy as np

def foo(arg0, arg1, arg2, arg3, arg4, arg5, sentinel):
    # t0 = arg0
    t0 = arg0
    
    # t1 = torch.tanh(t0) 
    # Note: PyTorch tanh on int returns float. The bug report comment says dtype=int64, 
    # but standard behavior is float. We cast to float16 for the op, then back to int64 
    # to satisfy the subsequent fill_ operation logic which expects int-like container.
    t1 = tf.cast(tf.tanh(tf.cast(t0, tf.float16)), tf.int64)
    
    # t2 = arg1, t3 = arg2
    t2 = arg1
    t3 = arg2
    
    # t4 = t2 * t3
    t4 = t2 * t3
    
    # t5 = t1.clone(); t5.fill_(t4.item())
    # In TF, we create a tensor like t1 and fill it with the scalar value of t4
    t5 = tf.fill(tf.shape(t1), tf.cast(t4[0], tf.int64)) # t4 is scalar-like (0-d tensor)
    
    # t6, t7, t8 are float16 inputs
    t6 = arg3
    t7 = arg4
    t8 = arg5
    
    # t9 = torch.cat([t6, t6, t7, t8], dim=2)
    t9 = tf.concat([t6, t6, t7, t8], axis=2)
    
    # --- ADAPTATION START ---
    # Original: t10 = t9.std(dim=2) -> Shape (256, 88)
    # Adaptation: Use tf.keras.ops.outer
    # To maintain shape compatibility for the subsequent embedding lookup (which expects (256, 88) weights),
    # we perform an outer product on slices of the last dimension.
    # We take the first batch item to keep computation manageable but valid.
    # t9 shape is (256, 88, 4). 
    # We take t9[0] -> (88, 4).
    # We perform outer product on the first two vectors of size 88.
    
    t9_slice = t9[0] # Shape (88, 4)
    vec_a = t9_slice[:, 0] # Shape (88,)
    vec_b = t9_slice[:, 1] # Shape (88,)
    
    # Call the similar API: tf.keras.ops.outer
    # Result shape: (88, 88)
    t10 = tf.keras.ops.outer(vec_a, vec_b)
    # --- ADAPTATION END ---
    
    # t11 = torch.nn.functional.embedding(torch.clamp(t5, 0, t10.size(0) - 1).to(torch.long), t10)
    # t10.size(0) is 88.
    # Clamp indices to [0, 87]
    indices = tf.clip_by_value(t5, 0, tf.shape(t10)[0] - 1)
    t11 = tf.nn.embedding_lookup(t10, indices)
    
    # output = t11 + sentinel
    output = t11 + sentinel
    return output

# Input generation
# arg0: int64, [47]
arg0 = tf.constant(np.random.randint(0, 1000, [47]), dtype=tf.int64)
# arg1, arg2: int64, []
arg1 = tf.constant(np.random.randint(0, 1000, []), dtype=tf.int64)
arg2 = tf.constant(np.random.randint(0, 1000, []), dtype=tf.int64)
# arg3, arg4, arg5: float16, [256, 88, 1], requires_grad=True
# We use tf.Variable to track gradients
arg3 = tf.Variable(np.random.rand(256, 88, 1).astype(np.float16))
arg4 = tf.Variable(np.random.rand(256, 88, 1).astype(np.float16))
arg5 = tf.Variable(np.random.rand(256, 88, 1).astype(np.float16))
# sentinel: float16, requires_grad=True
sentinel = tf.Variable(0.0, dtype=tf.float16)

if __name__ == '__main__':
    # 1. Test Eager Mode
    with tf.GradientTape() as tape:
        out_eager = foo(arg0, arg1, arg2, arg3, arg4, arg5, sentinel)
    grads_eager = tape.gradient(out_eager, [arg3, arg4, arg5, sentinel])
    
    # Basic checks
    assert out_eager.dtype == tf.float16, "Eager output dtype mismatch"
    assert grads_eager[0] is not None, "Eager gradient is None"
    print('Eager Success! ')

    # 2. Test Compiled Mode (tf.function)
    # This is analogous to torch.compile
    compiled_foo = tf.function(foo)
    
    with tf.GradientTape() as tape:
        out_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4, arg5, sentinel)
    grads_compiled = tape.gradient(out_compiled, [arg3, arg4, arg5, sentinel])
    
    # Basic checks
    assert out_compiled.dtype == tf.float16, "Compiled output dtype mismatch"
    assert grads_compiled[0] is not None, "Compiled gradient is None"
    
    # Check for divergence (values should be same)
    if not np.allclose(out_eager.numpy(), out_compiled.numpy()):
        print("Warning: Eager and Compiled outputs diverge.")
    else:
        print('Compile Success! ')
import torch
import tensorflow as tf
import numpy as np
import sys

def foo(arg0, arg1, arg2, arg3, arg4):
    # Replicating the structure of the original PyTorch test case
    # Original: t0 = arg0 (5, 4)
    t0 = arg0
    
    # Original: t1 = arg1 (5, 1024), t2 = arg2 (1024, 4)
    # Original: t3 = torch.addmm(t0, t1, t2) -> t0 + (t1 @ t2)
    # We use tf.stop_gradient on the matrix multiplication part to test the API
    # Note: tf.stop_gradient is an identity op in the forward pass.
    matmul_res = tf.matmul(arg1, arg2)
    stopped_matmul = tf.stop_gradient(matmul_res)
    t3 = t0 + stopped_matmul
    
    # Original: t4 = t3.norm()
    t4 = tf.norm(t3)
    
    # Original: t5 = arg3 (3, 4, 5, 2)
    t5 = arg3
    # Original: t6 = t5.var(dim=0)
    t6 = tf.math.reduce_variance(t5, axis=0)
    # Original: t7 = t6.var()
    t7 = tf.math.reduce_variance(t6)
    
    # Original: t8 = arg4, t9 = relu(t8)
    t8 = arg4
    t9 = tf.nn.relu(t8)
    
    # Original: t10 = t7 + t4 + t9
    t10 = t7 + t4 + t9
    
    # Original: t11 = torch.pow(torch.pow(t4, t7), t10)
    # Using tf.pow for the power operation
    t11 = tf.pow(tf.pow(t4, t7), t10)
    
    output = t11
    return output

# Generate inputs matching the shapes of the original PyTorch test case
# Using float32 for broader compatibility compared to bfloat16
arg0 = tf.random.normal([5, 4], dtype=tf.float32)
arg1 = tf.random.normal([5, 1024], dtype=tf.float32)
arg2 = tf.random.normal([1024, 4], dtype=tf.float32)
arg3 = tf.random.normal([3, 4, 5, 2], dtype=tf.float32)
arg4 = tf.random.normal([], dtype=tf.float32)

if __name__ == '__main__':
    # 1. Run Eagerly
    out_eager = foo(arg0, arg1, arg2, arg3, arg4)
    print('Eager Success! ')

    # 2. Run Compiled (tf.function)
    # Note: In TensorFlow, tf.function is the equivalent of torch.compile
    compiled_foo = tf.function(foo)
    out_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4)
    print('Compile Success! ')

    # 3. Compare outputs
    # The original bug reported an 8% difference. We check for significant divergence.
    # Since tf.stop_gradient is a forward-pass identity, results should be identical.
    
    # Convert to numpy for absolute comparison if needed, or use tf ops
    diff = tf.abs(out_eager - out_compiled)
    
    # Use a small tolerance for floating point operations
    tolerance = 1e-5
    if diff < tolerance:
        print(f'Relative diff check passed (diff < {tolerance})')
    else:
        print(f' Outputs differ significantly!')
        print('out_eager:', out_eager.numpy())
        print('out_compiled:', out_compiled.numpy())
        print('Absolute diff:', diff.numpy())
        sys.exit(1)
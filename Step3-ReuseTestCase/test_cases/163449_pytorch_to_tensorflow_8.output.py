import torch
import tensorflow as tf
import numpy as np
import sys

def foo(arg0, arg1, arg2, arg3, arg4):
    # Use the specific API requested: tf.compat.v1.convert_to_tensor
    # to handle the conversion of inputs to tensors.
    t0 = tf.compat.v1.convert_to_tensor(arg0, dtype=tf.bfloat16)
    t1 = tf.compat.v1.convert_to_tensor(arg1, dtype=tf.bfloat16)
    t2 = tf.compat.v1.convert_to_tensor(arg2, dtype=tf.bfloat16)
    t5 = tf.compat.v1.convert_to_tensor(arg3, dtype=tf.float32)
    t8 = tf.compat.v1.convert_to_tensor(arg4, dtype=tf.float32)

    # Replicate the logic from the PyTorch bug report
    # t3 = torch.addmm(t0, t1, t2) -> t0 + (t1 @ t2)
    t3 = tf.add(tf.matmul(t1, t2), t0)

    # t4 = t3.norm()
    # Note: PyTorch norm on bfloat16 returns bfloat16. 
    # TF norm promotes to float32, so we cast back to match the original trace's dtype.
    t4 = tf.cast(tf.norm(t3), dtype=tf.bfloat16)

    # t6 = t5.var(dim=0)
    t6 = tf.math.reduce_variance(t5, axis=0)

    # t7 = t6.var()
    t7 = tf.math.reduce_variance(t6)

    # t9 = torch.nn.functional.relu(t8)
    t9 = tf.nn.relu(t8)

    # t10 = t7 + t4 + t9
    # t4 is bfloat16, t7 and t9 are float32. Result is float32.
    t10 = t7 + t4 + t9

    # t11 = torch.pow(torch.pow(t4, t7), t10)
    # t4 is bfloat16, t7 is float32. pow promotes to float32.
    t11 = tf.pow(tf.pow(tf.cast(t4, tf.float32), t7), t10)

    return t11

# Generate inputs using numpy arrays
# Numpy does not support bfloat16 natively, so we use float32 and let convert_to_tensor handle the cast.
np.random.seed(42)
arg0 = np.random.rand(5, 4).astype(np.float32)
arg1 = np.random.rand(5, 1024).astype(np.float32)
arg2 = np.random.rand(1024, 4).astype(np.float32)
arg3 = np.random.rand(3, 4, 5, 2).astype(np.float32)
arg4 = np.random.rand([]).astype(np.float32)

if __name__ == '__main__':
    # Eager execution
    out_eager = foo(arg0, arg1, arg2, arg3, arg4)
    print('Eager Success! ')

    # Compiled execution (tf.function is the TF equivalent of torch.compile)
    compiled_foo = tf.function(foo)
    out_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4)
    print('Compile Success! ')

    # Compare outputs
    out_eager_val = out_eager.numpy()
    out_compiled_val = out_compiled.numpy()

    diff = np.abs(out_eager_val - out_compiled_val)
    rel_diff = diff / (np.abs(out_eager_val) + 1e-12) * 100

    print(f'Relative diff: {rel_diff:.6f}%')
    
    # Check for significant divergence similar to the original bug report
    if rel_diff > 5:
        print(f' Outputs differ significantly!')
        print('out_eager:', out_eager_val)
        print('out_compiled:', out_compiled_val)
        sys.exit(1)
    else:
        print('Test Passed: No significant divergence detected.')
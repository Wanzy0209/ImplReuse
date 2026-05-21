import torch
import tensorflow as tf
import numpy as np
import sys

def foo(arg0, arg1, arg2, arg3, arg4):
    # Use the similar API: tf.keras.ops.convert_to_tensor
    # to handle the input arguments, mirroring the structure of the original test.
    t0 = tf.keras.ops.convert_to_tensor(arg0)
    t1 = tf.keras.ops.convert_to_tensor(arg1)
    t2 = tf.keras.ops.convert_to_tensor(arg2)
    t3 = tf.keras.ops.convert_to_tensor(arg3)
    t4 = tf.keras.ops.convert_to_tensor(arg4)

    # Replicate the logic from the PyTorch bug report.
    # Original: t3 = torch.addmm(t0, t1, t2) -> t0 + t1 @ t2
    # Note: We use float32 here to ensure the test runs on standard CPU/GPU without 
    # bfloat16 specific hardware requirements, though the original bug was triggered 
    # by bfloat16 precision.
    matmul_res = tf.linalg.matmul(t1, t2)
    t3_res = t0 + matmul_res

    # t4 = t3.norm()
    t4_norm = tf.norm(t3_res)

    # t5 = arg3
    t5 = t3

    # t6 = t5.var(dim=0)
    # PyTorch var default is unbiased (Bessel's correction). TF reduce_variance is also unbiased.
    t6 = tf.math.reduce_variance(t5, axis=0)

    # t7 = t6.var()
    t7 = tf.math.reduce_variance(t6)

    # t8 = arg4
    t8 = t4

    # t9 = torch.nn.functional.relu(t8)
    t9 = tf.nn.relu(t8)

    # t10 = t7 + t4 + t9
    t10 = t7 + t4_norm + t9

    # t11 = torch.pow(torch.pow(t4, t7), t10)
    t11 = tf.pow(tf.pow(t4_norm, t7), t10)

    return t11

# Setup inputs
# Using float32 for compatibility. Original used bfloat16.
arg0 = np.random.rand(5, 4).astype(np.float32)
arg1 = np.random.rand(5, 1024).astype(np.float32)
arg2 = np.random.rand(1024, 4).astype(np.float32)
arg3 = np.random.rand(3, 4, 5, 2).astype(np.float32)
arg4 = np.random.rand(1).astype(np.float32)[0] # Scalar

if __name__ == '__main__':
    # Eager execution
    out_eager = foo(arg0, arg1, arg2, arg3, arg4)
    print('Eager Success! ')

    # Compiled execution (tf.function is the TF equivalent of torch.compile)
    compiled_foo = tf.function(foo)
    out_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4)
    print('Compile Success! ')

    # Compare outputs (forward)
    # The original bug checked relative difference > 5%
    out_eager_val = out_eager.numpy()
    out_compiled_val = out_compiled.numpy()
    
    diff = np.abs(out_eager_val - out_compiled_val)
    rel_diff = diff / (np.abs(out_eager_val) + 1e-12) * 100
    
    print(f'Relative diff: {rel_diff:.6f}%')
    
    # In TF, eager and graph mode usually match perfectly for float32.
    # If we were using bfloat16, we might see divergence similar to the PyTorch bug.
    # We assert they are close to verify the API (convert_to_tensor) doesn't introduce issues.
    if rel_diff > 5:
        print(f' Forward outputs differ significantly (relative)!')
        print('out_eager:', out_eager_val)
        print('out_compiled:', out_compiled_val)
        print('Absolute diff:', diff)
        print('Relative diff (%):', rel_diff)
        sys.exit(1)
    else:
        print('Test Passed! ')
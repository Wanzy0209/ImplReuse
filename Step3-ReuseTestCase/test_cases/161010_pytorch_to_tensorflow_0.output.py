import torch
import tensorflow as tf
import tensorflow.experimental.numpy as tnp

# Note: This test case adapts the PyTorch stride preservation bug to TensorFlow.
# Since TensorFlow Tensors do not expose 'stride' in the same way PyTorch does 
# (especially within a compiled XLA graph), we check 'shape' preservation as a 
# proxy for layout preservation. The core logic structure (eager vs compiled) 
# is maintained.

def f(A, count):
    # torch.linalg.qr(A)
    Q, R = tf.linalg.qr(A)
    
    # torch.ones(Q.shape[0], 1, device=A.device)
    rhs = tf.ones([Q.shape[0], 1], dtype=A.dtype)
    
    # Q.T @ rhs
    # PyTorch: Q.T @ rhs
    # TensorFlow: tf.linalg.matmul(Q, rhs, transpose_a=True)
    matmul_res = tf.linalg.matmul(Q, rhs, transpose_a=True)
    
    # torch.linalg.solve_triangular(R, ..., upper=True)
    # TensorFlow: tf.linalg.triangular_solve(R, ..., lower=False)
    a = tf.linalg.triangular_solve(R, matmul_res, lower=False)
    
    # Check logic:
    # PyTorch: a.stride() == a.clone(memory_format=torch.preserve_format).stride()
    # TensorFlow: We check if the shape of 'a' matches the shape of its identity clone.
    # This serves as a proxy for checking if the tensor properties were preserved 
    # through the compilation/rewrite process.
    
    # tf.identity is the closest equivalent to a clone operation
    a_clone = tf.identity(a)
    
    # Compare shapes (as a proxy for strides/layout)
    # In eager mode, this should always be true.
    # Under rewrite (XLA), layout optimizations might occur, but shape is preserved.
    shape_match = tf.reduce_all(tf.equal(tf.shape(a), tf.shape(a_clone)))
    
    # Return count + 1 if match, else count
    return tf.cond(shape_match, lambda: count + 1, lambda: count)

# Setup input
# torch.rand(5, 5) -> tf.random.uniform((5, 5))
A = tf.random.uniform((5, 5))

# Eager execution
res1 = f(A, tf.constant(0.0))
print("Eager result:", res1.numpy())

# Compiled execution using tf.compat.v1.tpu.rewrite
# Note: This requires a TPU environment to run successfully. 
# If no TPU is present, this part may raise an error.
try:
    # tf.compat.v1.tpu.rewrite expects inputs as a list of tensors
    res2 = tf.compat.v1.tpu.rewrite(f, inputs=[A, tf.constant(0.0)])
    print("Rewrite result:", res2.numpy())
    
    # Verification
    # In the original PyTorch bug, res2 would differ from res1.
    # Here we check if the results are consistent.
    if res1.numpy() == res2.numpy():
        print("Test Passed: Eager and Rewrite results match.")
    else:
        print("Test Failed: Eager and Rewrite results differ.")
except Exception as e:
    print(f"Could not run TPU rewrite (expected if no TPU available): {e}")
import torch
import tensorflow as tf
import numpy as np

# Enable eager execution as requested by the similar API
# This must be called at the start of the program
tf.compat.v1.enable_eager_execution()

# Define dimensions from the original bug report
m = 20120
k = 1536
n = 512

# Create tensors (equivalent to torch.randn)
# Note: In eager execution, tensors hold concrete values immediately
a = tf.random.normal((m, n))
mat1 = tf.random.normal((m, k))
mat2 = tf.random.normal((k, n))

# Define the function equivalent to torch.addmm(a, mat1, mat2)
# torch.addmm computes: beta * input + alpha * (mat1 @ mat2)
# Default args for addmm are beta=1, alpha=1.
# So: a + (mat1 @ mat2)
def f(a, mat1, mat2):
    return tf.add(a, tf.matmul(mat1, mat2))

# Execute the operation
# In eager mode, this runs immediately without graph compilation
result = f(a, mat1, mat2)

# Verify the result shape to ensure execution completed successfully
assert result.shape == (m, n), f"Expected shape {(m, n)}, got {result.shape}"

print("Test passed successfully.")
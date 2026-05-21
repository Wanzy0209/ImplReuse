import torch
import tensorflow as tf
import numpy as np

# Set seed for reproducibility
tf.random.set_seed(2025)

# Define the function using TensorFlow operations
# Note: To mimic in-place operations (sin_) and index assignment (y[2]=...),
# we must use tf.Variable.
def foo(x_var):
    # x[0].sin_()
    x_var[0].assign(tf.math.sin(x_var[0]))
    # x[1].sin_()
    x_var[1].assign(tf.math.sin(x_var[1]))

    # y = torch.zeros_like(x)
    y = tf.Variable(tf.zeros_like(x_var))

    # y[2] = x[0]
    y[2].assign(x_var[0])
    # y[3] = x[1]
    y[3].assign(x_var[1])

    return y

# Prepare inputs
# PyTorch: torch.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], dtype=torch.float32)
data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
x = tf.Variable(data, dtype=tf.float32)
cx = tf.Variable(data, dtype=tf.float32)

# Run baseline
res = foo(x)

# Run inside the target API: tf.compat.v1.name_scope
# This mimics the structure of applying the API to the execution flow.
with tf.compat.v1.name_scope("test_scope"):
    cres = foo(cx)

# Verify behavior
# In the PyTorch bug, cres would differ (double sin).
# Here we verify that name_scope allows correct execution.
np.testing.assert_allclose(res.numpy(), cres.numpy())
print("Test passed.")
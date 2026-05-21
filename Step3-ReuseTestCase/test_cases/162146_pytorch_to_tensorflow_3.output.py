import torch
import tensorflow as tf
import numpy as np

# The API to test: enable_eager_execution
# This ensures operations are executed immediately, similar to the default PyTorch behavior
# (as opposed to graph mode which is somewhat analogous to torch.compile)
tf.compat.v1.enable_eager_execution()

def foo(x):
    # Mimic in-place operations (x[0].sin_())
    # In TensorFlow, we use tf.Variable and assign to mimic in-place behavior
    x[0].assign(tf.sin(x[0]))
    x[1].assign(tf.sin(x[1]))
    
    # Create y
    y = tf.zeros_like(x)
    
    # Mimic index put (y[2] = x[0])
    # tf.tensor_scatter_nd_update is used to update specific indices of a tensor
    y = tf.tensor_scatter_nd_update(y, [[2]], [x[0]])
    y = tf.tensor_scatter_nd_update(y, [[3]], [x[1]])
    
    return y

# Setup input data
np_data = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], dtype=np.float32)
x_var = tf.Variable(np_data)

# Execute the function
res = foo(x_var)

# Calculate expected result manually
# x[0] becomes sin([1,2,3])
# x[1] becomes sin([4,5,6])
# y[2] = x[0], y[3] = x[1], rest are 0
expected_x = np_data.copy()
expected_x[0] = np.sin(expected_x[0])
expected_x[1] = np.sin(expected_x[1])
expected_y = np.zeros_like(expected_x)
expected_y[2] = expected_x[0]
expected_y[3] = expected_x[1]

# Verify the behavior
try:
    np.testing.assert_allclose(res.numpy(), expected_y)
    print("Test passed: TensorFlow eager execution behaves as expected.")
except AssertionError as e:
    print(f"Test failed: {e}")
import torch
import tensorflow as tf
import numpy as np

# Define the function using the requested API: tf.keras.backend.name_scope
# We use tf.function to mimic torch.compile behavior (graph mode)
@tf.function
def foo_compiled(x_var):
    with tf.keras.backend.name_scope("inplace_ops"):
        # x[0].sin_()
        # In TensorFlow, Tensors are immutable. We use tf.Variable for in-place operations.
        x_var[0].assign(tf.math.sin(x_var[0]))
        # x[1].sin_()
        x_var[1].assign(tf.math.sin(x_var[1]))

    with tf.keras.backend.name_scope("index_put"):
        # y = torch.zeros_like(x)
        y_var = tf.Variable(tf.zeros_like(x_var))
        # y[2] = x[0]
        y_var[2].assign(x_var[0])
        # y[3] = x[1]
        y_var[3].assign(x_var[1])
        return y_var

# Eager version for comparison
def foo_eager(x_var):
    with tf.keras.backend.name_scope("inplace_ops"):
        x_var[0].assign(tf.math.sin(x_var[0]))
        x_var[1].assign(tf.math.sin(x_var[1]))

    with tf.keras.backend.name_scope("index_put"):
        y_var = tf.Variable(tf.zeros_like(x_var))
        y_var[2].assign(x_var[0])
        y_var[3].assign(x_var[1])
        return y_var

# Setup data
# Note: We use tf.Variable because TF Tensors are immutable, unlike PyTorch Tensors
# which support in-place operations.
data = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], dtype=np.float32)
x = tf.Variable(data)
cx = tf.Variable(data)

# Run eager
res = foo_eager(x)

# Run compiled (graph mode)
cres = foo_compiled(cx)

# Assert
# In TF, eager and graph mode should generally produce the same results for this logic.
# This test verifies that the logic holds when wrapped in name_scope and compiled.
np.testing.assert_allclose(res.numpy(), cres.numpy())
print("Test passed.")
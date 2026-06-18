import torch
import tensorflow as tf
import numpy as np

# Enable eager execution (The API under test)
# This is the TensorFlow equivalent to ensuring operations run immediately,
# analogous to the eager mode check in the PyTorch bug report.
tf.compat.v1.enable_eager_execution()

# Set seed for reproducibility
tf.random.set_seed(13653)

def fuzzed_program(arg_0, arg_1, sentinel):
    # var_node_3 = torch.full((), 1.0, dtype=torch.float32)
    var_node_3 = tf.constant(1.0, dtype=tf.float32)

    # var_node_2 = var_node_3.item()
    # In TensorFlow eager, we access the scalar value via .numpy()
    var_node_2 = var_node_3.numpy().item()

    var_node_5 = -3
    var_node_6 = arg_0
    var_node_4 = var_node_5 + var_node_6

    var_node_1 = var_node_2 + var_node_4

    var_node_9 = 1
    var_node_10 = -10
    # PyTorch comment indicates int64, but operator '/' implies float division.
    # We follow the operator semantics.
    var_node_8 = var_node_9 / var_node_10

    var_node_12 = arg_1
    var_node_13 = -5
    var_node_11 = var_node_12 / var_node_13

    var_node_7 = var_node_8 + var_node_11

    var_node_0 = var_node_1 * var_node_7

    # result = var_node_0 * sentinel
    result = var_node_0 * sentinel

    # if result.is_complex(): result = result.real
    # Check if the result tensor is complex
    if result.dtype.is_complex:
        result = tf.math.real(result)

    return result

# Sentinel tensor to ensure gradient computation context
# PyTorch uses requires_grad=True. In TF, we use a Variable.
sentinel = tf.Variable(1.0, dtype=tf.float32)

# Generate arguments similar to the original test case
# torch.randn(()).item() generates a random float, then cast to int
arg_0 = int(tf.random.normal(()).numpy().item())
arg_1 = int(tf.random.normal(()).numpy().item())

args = (arg_0, arg_1, sentinel)

# Run the program
result_original = fuzzed_program(*args)
print(' eager success')

# Note: The original bug was a divergence between eager and compiled mode.
# In TensorFlow, enable_eager_execution ensures we stay in eager mode.
# We verify the logic executes correctly.
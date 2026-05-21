import torch
import tensorflow as tf
import numpy as np

# Set the random seed for reproducibility
tf.random.set_seed(42)

# Define the condition and body for the while_loop
# This mimics the logic of the PyTorch function 'fn'
def cond(i, x):
    # Run the loop body once to match the single execution in the PyTorch example
    return i < 1

def body(i, x):
    # PyTorch logic: return x * torch.sigmoid(torch.randn(1, device="cuda"))
    # TensorFlow logic: x * tf.sigmoid(tf.random.normal(...))
    return i + 1, x * tf.sigmoid(tf.random.normal(shape=tf.shape(x)))

# 1. Eager Execution
# Reset seed to ensure consistent RNG state
tf.random.set_seed(42)

# Initialize input variable
eager_in = tf.Variable(tf.ones((1,), dtype=tf.float32))

# Execute in eager mode
with tf.GradientTape() as tape:
    # loop_vars: [iteration_counter, input_tensor]
    _, eager_out = tf.keras.ops.while_loop(cond, body, loop_vars=[tf.constant(0), eager_in])

eager_in_grad = tape.gradient(eager_out, eager_in)

# 2. Graph Execution (Equivalent to CUDA Graph capture)
# In TensorFlow, tf.function captures the execution graph (similar to torch.cuda.graph)
@tf.function
def graph_op(x):
    _, out = tf.keras.ops.while_loop(cond, body, loop_vars=[tf.constant(0), x])
    return out

# Reset seed to ensure consistent RNG state for the graph run
tf.random.set_seed(42)

# Initialize input variable for graph run
graph_in = tf.Variable(tf.ones((1,), dtype=tf.float32))

# Execute in graph mode
with tf.GradientTape() as tape:
    graph_out = graph_op(graph_in)

graph_in_grad = tape.gradient(graph_out, graph_in)

# 3. Assertion
# Verify that gradients from eager execution match graph execution
# This checks if the API behaves consistently under graph capture
assert tf.reduce_all(tf.abs(eager_in_grad - graph_in_grad) < 1e-5), "Mismatch in gradient outputs"

print("Eager Grad:", eager_in_grad.numpy())
print("Graph Grad:", graph_in_grad.numpy())
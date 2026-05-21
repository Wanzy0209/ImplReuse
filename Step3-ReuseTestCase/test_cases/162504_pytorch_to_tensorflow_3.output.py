import torch
import tensorflow as tf
import numpy as np

# Define the loop condition and body
def cond(i, x):
    return i < 2

def body(i, x):
    # Mimic the operation: x * sigmoid(...)
    # Using a deterministic function of x to ensure gradient consistency
    return i + 1, x * tf.math.sigmoid(x + 1.0)

# Initialize / Warmup (similar to fn(torch.ones(...)) in PyTorch)
# In TF, tracing the function once helps initialize resources
dummy_in = tf.constant(1.0)
i_dummy = tf.constant(0)
tf.while_loop(cond, body, [i_dummy, dummy_in])

# 1. Eager Execution
eager_in = tf.constant(1.0)
with tf.GradientTape() as tape:
    i_eager = tf.constant(0)
    _, eager_out = tf.while_loop(cond, body, [i_eager, eager_in])
eager_in_grad = tape.gradient(eager_out, eager_in)

# 2. Graph Execution (tf.function)
# This corresponds to the 'with torch.cuda.graph(g):' block
@tf.function
def graph_fn(x):
    i = tf.constant(0)
    _, out = tf.while_loop(cond, body, [i, x])
    return out

graph_in = tf.constant(1.0)
with tf.GradientTape() as tape:
    graph_out = graph_fn(graph_in)
graph_in_grad = tape.gradient(graph_out, graph_in)

# 3. Verification
print(f"Eager Grad: {eager_in_grad.numpy()}")
print(f"Graph Grad: {graph_in_grad.numpy()}")

# Using a small tolerance for floating point comparison, though exact match is expected for this simple op
assert np.allclose(eager_in_grad.numpy(), graph_in_grad.numpy(), rtol=1e-5, atol=1e-5), "Mismatch in gradient outputs"
print("Test Passed: Gradients match between eager and graph execution.")
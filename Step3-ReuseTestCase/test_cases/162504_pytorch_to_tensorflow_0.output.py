import torch
import tensorflow as tf
import numpy as np

# Set global random seed for reproducibility
tf.random.set_seed(42)

# Define the map function similar to the PyTorch 'fn'
# Note: parallel_interleave expects map_func to return a Dataset
def map_func(x):
    # Mimic the random operation in the original bug report
    # Using stateful random op to check consistency between eager and graph modes
    noise = tf.random.normal(shape=[])
    val = x * tf.math.sigmoid(noise)
    return tf.data.Dataset.from_tensors(val)

# Input data
# Mimicking torch.ones(1, device="cuda")
input_data = [1.0, 1.0, 1.0]
ds = tf.data.Dataset.from_tensor_slices(input_data)

# 1. Eager Execution
# Apply parallel_interleave
# sloppy=False ensures deterministic order, similar to preserving RNG state logic
eager_ds = ds.apply(tf.data.experimental.parallel_interleave(
    map_func,
    cycle_length=2,
    sloppy=False,
    block_length=1
))
eager_results = list(eager_ds.as_numpy_iterator())

# Reset seed to ensure same starting state for the graph run
tf.random.set_seed(42)

# 2. Graph Execution (tf.function)
# In TensorFlow, tf.function is the equivalent of capturing a computation graph
@tf.function
def run_in_graph():
    graph_ds = ds.apply(tf.data.experimental.parallel_interleave(
        map_func,
        cycle_length=2,
        sloppy=False,
        block_length=1
    ))
    # Reduce the dataset to a tensor to return it from the function
    return graph_ds.reduce(tf.constant([], dtype=tf.float32), lambda state, x: tf.concat([state, [x]], axis=0))

graph_results = run_in_graph().numpy()

# 3. Assertion
# Check if the outputs match (mimicking the gradient check in the original, checking consistency)
# The original bug checks if gradients match; here we check if the data pipeline outputs match
# because tf.data pipelines do not produce gradients in the same way tensor ops do.
try:
    assert np.allclose(eager_results, graph_results, rtol=0.0, atol=0.0), "Mismatch in outputs between eager and graph execution"
    print("Eager results:", eager_results)
    print("Graph results:", graph_results)
    print("Test passed.")
except AssertionError as e:
    print(e)
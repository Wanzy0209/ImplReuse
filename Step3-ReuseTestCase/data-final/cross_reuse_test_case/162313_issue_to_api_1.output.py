import torch
import tensorflow as tf

# Helper function to simulate a graph break (forcing execution back to Python)
def _simulate_graph_break(x):
    return x

# Set initial state (mimics flag = True)
tf.config.experimental.enable_tensor_float_32_execution(True)

@tf.function
def fn(x):
    # Use the similar API to determine the execution path
    # This mirrors the 'if flag:' check in the original bug report
    if tf.config.experimental.tensor_float_32_execution_enabled():
        # Branch 1
        # Simulate torch._dynamo.graph_break() using tf.py_function
        x = tf.py_function(_simulate_graph_break, [x], tf.float32)
    else:
        # Branch 2
        # Simulate torch._dynamo.graph_break() using tf.py_function
        x = tf.py_function(_simulate_graph_break, [x], tf.float32)
    
    return x + 4

# First execution
result1 = fn(tf.ones(3))

# Toggle the global state (mimics flag = False)
tf.config.experimental.enable_tensor_float_32_execution(False)

# Second execution
# In the original bug, this pattern (state change + re-entry with graph breaks)
# triggered a KeyError in the resume function creation logic.
# This test verifies that the TensorFlow equivalent handles the state transition
# and graph boundaries correctly.
result2 = fn(tf.ones(3))

# Assertions to verify correct behavior
assert result1.numpy()[0] == 5.0
assert result2.numpy()[0] == 5.0
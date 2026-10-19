import torch
import tensorflow as tf

# Original logic: inner function
def inner(x):
    return x + 1

# Original logic: compiled function (tf.function is the equivalent of torch.compile)
# We leverage tf.control_dependencies as requested by the similar API pattern.
# This helps manage execution flow and realization of variables, which relates
# to the original issue about tracking realized variables.
@tf.function
def fn(x):
    # Calculate the first step
    y = inner(x)
    
    # Use control_dependencies to ensure the variable is realized/controlled
    # before the next operation, mimicking the debug context requested in the issue.
    with tf.control_dependencies([y]):
        x = y
        return inner(x)

# Replacement: torch.ones -> torch.linspace
# Original: torch.ones(3) -> shape (3,)
# TF equivalent: tf.linspace(0.0, 1.0, 3)
x = tf.linspace(0.0, 1.0, 3)

# Execute
result = fn(x)

# Verify result
# Input: [0, 0.5, 1]
# Step 1: [1, 1.5, 2]
# Step 2: [2, 2.5, 3]
expected = tf.constant([2.0, 2.5, 3.0])
assert tf.reduce_all(tf.equal(result, expected)).numpy()
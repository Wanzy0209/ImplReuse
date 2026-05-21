import torch
import tensorflow as tf
import numpy as np

# Adapted test case for tf.keras.backend.set_value
# This test mirrors the structure of the original issue (Issue 162858) where
# an API is called within a compiled/graph context.
# Original: torch._dynamo.graph_break() inside @torch.compile
# Adapted: tf.keras.backend.set_value() inside @tf.function

# Create a variable to manipulate
my_var = tf.Variable([0.0, 0.0])

@tf.function
def fn(x):
    # Reproduce the arithmetic logic from the original issue
    y = x + 1
    z = x + y
    
    # Use the similar API to modify state
    # In the original issue, this was torch._dynamo.graph_break()
    # Here we use tf.keras.backend.set_value to interact with the graph
    new_value = np.array([99.0, 99.0])
    tf.keras.backend.set_value(my_var, new_value)
    
    return z

# Execute the function
input_tensor = tf.ones(2)
result = fn(input_tensor)

# Verify the behavior
# The original issue was about logging context during the break.
# Here we verify the side-effect of the API call in the graph context.
assert tf.reduce_all(my_var == tf.constant([99.0, 99.0])).numpy(), "Test failed: Variable not updated in graph mode"

print("Test passed: tf.keras.backend.set_value works correctly inside tf.function")
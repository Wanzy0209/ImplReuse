import torch
import tensorflow as tf

# Define a function that relies on the global backend configuration (epsilon)
# This mirrors the structure of the original recursive function 'fn' in the bug report,
# but uses the similar API (epsilon) instead of recursion logic.
@tf.function
def fn(x):
    # Access the global epsilon value, similar to how the original code 
    # implicitly relies on the recursion limit.
    return x + tf.keras.backend.epsilon()

# Set a global configuration value, analogous to sys.setrecursionlimit in the bug.
# We set it to a distinct value to verify it is picked up by the compiled function.
tf.keras.backend.set_epsilon(1.0)

# Execute the compiled function.
# In the original bug, this failed because the recursion limit wasn't respected by torch.compile.
# Here we test if tf.function respects the global epsilon setting.
input_tensor = tf.constant(0.0)
result = fn(input_tensor)

# Assertion to verify the behavior.
# If the compiled function respects the global setting, the result should be 1.0.
assert result.numpy() == 1.0, "The compiled function did not respect the global epsilon setting."

# Reset to default to avoid side effects
tf.keras.backend.set_epsilon(1e-07)
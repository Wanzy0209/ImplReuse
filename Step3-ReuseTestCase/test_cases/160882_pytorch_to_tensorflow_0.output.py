import torch
import tensorflow as tf

# Define the function using the similar API
# We use @tf.function to mimic the compilation behavior of torch.compile
@tf.function
def f(x: tf.Tensor) -> tf.Tensor:
    # Using tf.compat.v1.nn.dropout as the target API
    # Note: We provide a seed to ensure deterministic behavior in the graph
    return tf.compat.v1.nn.dropout(x, rate=0.5, seed=42)

B, F, T = 1, 641, 39

# Create source tensor
x_src = tf.random.normal((B, F, T))

# Create mismatched tensor by permuting dimensions (0, 2, 1)
# This mimics the shape change logic in the original bug report
x_mismatch = tf.transpose(x_src, perm=[0, 2, 1])

# First call (tracing/compilation happens here)
_ = f(x_src)

# Second call with different shape
# This verifies if the compiled function handles the shape mismatch gracefully
# (The original PyTorch bug crashed here)
_ = f(x_mismatch)

# Assertions to verify execution completed and shapes are as expected
assert x_src.shape == (1, 641, 39)
assert x_mismatch.shape == (1, 39, 641)
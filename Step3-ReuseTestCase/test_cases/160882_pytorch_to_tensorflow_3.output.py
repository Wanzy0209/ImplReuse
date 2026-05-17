import torch
import tensorflow as tf

# Adapted function using the target API: tf.nn.dropout
# The original function combined two tensors (real, imag) into a complex tensor.
# Here we adapt the logic to test shape handling during graph execution with dropout.
def f(x: tf.Tensor) -> tf.Tensor:
    return tf.nn.dropout(x, rate=0.5)

B, F, T = 1, 641, 39

# Create source tensor
# torch.randn(B, F, T) -> tf.random.normal((B, F, T))
x_src = tf.random.normal((B, F, T))

# Create mismatched tensor (permuted dimensions)
# r_src.permute(0, 2, 1) -> tf.transpose(x_src, [0, 2, 1])
x_mismatch = tf.transpose(x_src, [0, 2, 1])

# Compile the function
# torch.compile(f, fullgraph=True) -> tf.function(f)
compiled_f = tf.function(f)

# First call with original shape
# This traces the graph for the input shape (1, 641, 39)
try:
    _ = compiled_f(x_src)
    print("First call successful.")
except Exception as e:
    print(f"First call failed: {e}")

# Second call with mismatched shape
# In the original PyTorch bug, this caused an AssertionError because the compiled
# kernel expected the original input shapes. We verify if tf.nn.dropout handles
# the shape change gracefully or raises a similar error.
try:
    _ = compiled_f(x_mismatch)
    print("Second call successful.")
except Exception as e:
    print(f"Second call failed: {e}")
import torch
import tensorflow as tf
import tensorflow.experimental.numpy as tnp

# Define the function using the target API
def f(m: tf.Tensor) -> tf.Tensor:
    # Call the similar API: tf.experimental.numpy.triu
    return tnp.triu(m)

# Compile the function (equivalent to torch.compile)
# In TensorFlow, we use tf.function to create a graph/compiled version
compiled_f = tf.function(f)

# Dimensions from the original bug report
B, F, T = 1, 641, 39

# Create source tensor
src = tf.random.normal((B, F, T))

# Create mismatched tensor (permuted dimensions)
# Original shape: (1, 641, 39) -> Permuted: (1, 39, 641)
mismatch = tf.transpose(src, perm=[0, 2, 1])

# Run 1: Original shape
print("Running compiled function with original shape:", src.shape)
try:
    result1 = compiled_f(src)
    print("Result 1 shape:", result1.shape)
except Exception as e:
    print(f"Failed on original shape: {e}")

# Run 2: Permuted shape (The trigger for the PyTorch bug)
print("Running compiled function with permuted shape:", mismatch.shape)
try:
    result2 = compiled_f(mismatch)
    print("Result 2 shape:", result2.shape)
except Exception as e:
    print(f"Failed on permuted shape: {e}")
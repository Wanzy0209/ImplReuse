import torch
import tensorflow as tf

# Define eager function
def eager_fn(x, y):
    return tf.concat([x, y], axis=0)

# Define compiled function (similar to @torch.compile)
@tf.function
def compiled_fn(x, y):
    return tf.concat([x, y], axis=0)

# Create complex tensors
# PyTorch used shapes (1,) and (). tf.concat requires matching ranks.
# We adapt to (1,) and (0,) to test complex types with an empty tensor.
x = tf.constant([1.0 + 1.0j], dtype=tf.complex64)
y = tf.constant([], dtype=tf.complex64)

# Test eager
try:
    res_eager = eager_fn(x, y)
    print("eager success")
except Exception as e:
    print(f"eager failed: {e}")

# Test compiled
try:
    res_compiled = compiled_fn(x, y)
    print("compiler success")
except Exception as e:
    print(f"compiler failed: {e}")
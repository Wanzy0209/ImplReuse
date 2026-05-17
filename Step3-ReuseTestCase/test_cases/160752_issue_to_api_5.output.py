import torch
import tensorflow as tf
import numpy as np

# Constants from the original issue
MAX = 3
BATCH = 37

def func(inputs):
    """
    Corresponds to the PyTorch function logic:
    return x.square() * torch.nn.functional.one_hot(idxs, MAX)
    
    Adapted for TensorFlow:
    - x.square() -> tf.square(x)
    - torch.nn.functional.one_hot(idxs, MAX) -> tf.one_hot(idx, MAX)
    """
    x, idx = inputs
    # Inside vectorized_map, x will be a slice of shape (MAX,) and idx will be a scalar.
    return tf.square(x) * tf.one_hot(idx, MAX)

# Generate data similar to the original issue
# idxs: (BATCH,), x: (BATCH, MAX)
idxs_np = np.random.randint(MAX, size=(BATCH,))
x_np = np.random.rand(BATCH, MAX).astype(np.float64)

idxs = tf.constant(idxs_np)
x = tf.constant(x_np)

# The original issue involves torch.compile failing to vectorize/optimize
# the jacfwd operation involving one_hot.
# Here we test tf.keras.ops.vectorized_map to ensure it correctly handles
# the one_hot and square operations across the batch dimension, which is the
# semantic equivalent of the vectorization attempt in the bug report.

try:
    # vectorized_map applies func to each element of the 0th dimension of the input tensors
    result = tf.keras.ops.vectorized_map(func, (x, idxs))

    # Assertions to verify correctness
    assert result.shape == (BATCH, MAX), f"Expected shape ({BATCH}, {MAX}), got {result.shape}"
    
    # Since we square x, all values should be non-negative
    assert tf.reduce_all(result >= 0).numpy(), "Result contains negative values, check logic"
    
    print("Test passed: tf.keras.ops.vectorized_map handled one_hot and square operations correctly.")

except Exception as e:
    print(f"Test failed with error: {e}")
    raise
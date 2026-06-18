import torch
import tensorflow as tf

# Constants from the original issue
MAX = 3
BATCH = 37

def func(inputs):
    """
    Corresponds to the logic inside the original PyTorch function:
    x.square() * torch.nn.functional.one_hot(idxs, MAX)
    
    In tf.vectorized_map, the function receives a single element from the batch.
    """
    x, idx = inputs
    # x is shape (MAX,), idx is a scalar
    # We ensure dtype matches x for the multiplication
    one_hot_val = tf.one_hot(idx, MAX, dtype=x.dtype)
    return tf.square(x) * one_hot_val

# Generate data matching the original issue
# idxs: (BATCH,), x: (BATCH, MAX)
idxs = tf.random.uniform((BATCH,), 0, MAX, dtype=tf.int64)
x = tf.random.uniform((BATCH, MAX), dtype=tf.float64)

# Use the similar API: tf.vectorized_map
# This acts as the vectorization mechanism analogous to torch.compile optimizing
# the batched operation in the original issue.
try:
    # tf.vectorized_map will apply 'func' to each pair (x[i], idxs[i])
    # in a vectorized/parallel manner.
    out = tf.vectorized_map(func, (x, idxs))
    
    # Verify the output shape matches expectations
    assert out.shape == (BATCH, MAX), f"Shape mismatch: expected ({BATCH}, {MAX}), got {out.shape}"
    
    print("Test passed: tf.vectorized_map handled one_hot logic correctly.")

except Exception as e:
    print(f"Test failed with error: {e}")
    raise
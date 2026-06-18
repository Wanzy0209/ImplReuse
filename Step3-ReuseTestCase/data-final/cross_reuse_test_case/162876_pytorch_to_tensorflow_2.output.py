import torch
import tensorflow as tf
import numpy as np

# Adapted test case for tf.experimental.numpy.tril
# The original issue involved torch.aminmax returning a structseq that couldn't be instantiated
# from its representation. Here we verify tf.experimental.numpy.tril behavior.

# Create a 2D tensor (required for tril)
input_tensor = tf.constant([[1, -3, 5], [2, 4, 6]])

# Call the API
result = tf.experimental.numpy.tril(input_tensor)

# Verify the behavior
# The result should be a Tensor
assert isinstance(result, tf.Tensor), "Result should be a Tensor"

# The result should have the lower triangle preserved and upper triangle zeroed
expected = tf.constant([[1, 0, 0], [2, 4, 0]])

# Check equality
assert tf.reduce_all(tf.equal(result, expected)).numpy(), "Values do not match expected lower triangle"

print("Test passed.")
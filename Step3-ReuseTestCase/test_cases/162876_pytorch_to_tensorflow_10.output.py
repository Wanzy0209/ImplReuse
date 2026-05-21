import torch
import tensorflow as tf

# Adapted from the original PyTorch test case inputs
# Original: torch.tensor([1, -3, 5])
x = tf.constant([1, -3, 5])

# tf.compat.v1.math.multiply_no_nan requires two arguments (x, y).
# We define y to test the specific behavior of the API (handling zeros).
# Using y = [1, 0, 1] to verify that -3 * 0 results in 0 (not NaN).
y = tf.constant([1, 0, 1])

# Call the API
result = tf.compat.v1.math.multiply_no_nan(x, y)

# Verify the behavior
# Expected result: [1 * 1, -3 * 0, 5 * 1] = [1, 0, 5]
expected = tf.constant([1, 0, 5])

# Assert that the result matches the expected output
assert tf.reduce_all(tf.equal(result, expected)).numpy(), f"Test failed: Expected {expected.numpy()}, got {result.numpy()}"

print("Test passed. Result:", result.numpy())
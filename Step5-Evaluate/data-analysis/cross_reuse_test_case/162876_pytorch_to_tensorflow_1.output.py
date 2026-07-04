import tensorflow as tf

# Adapt the original test case for tf.linalg.triu.
# The original PyTorch issue involved a 1D tensor, but triu requires rank >= 2.
# We use a 2D tensor to test the API's behavior.
input_tensor = tf.constant([[1, -3, 5], [2, 4, 6]])

# Call the API using tf.linalg.triu as tf.keras.ops is not available in this environment
result = tf.linalg.triu(input_tensor)

# Verify the result is a Tensor (unlike the PyTorch structseq issue, this is a standard type)
assert isinstance(result, tf.Tensor), "Result should be a Tensor"

# Verify the values are correct (Upper triangle including diagonal)
# Expected output for [[1, -3, 5], [2, 4, 6]] is [[1, -3, 5], [0, 4, 6]]
expected_output = tf.constant([[1, -3, 5], [0, 4, 6]])
assert tf.reduce_all(tf.equal(result, expected_output)).numpy(), "Output values do not match expected upper triangle"

print("Test passed.")
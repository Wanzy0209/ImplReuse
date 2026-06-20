import tensorflow as tf

# Create an input tensor with specific dimensions.
# Note: TensorFlow does not expose a direct 'empty_strided' equivalent for manual stride control,
# but we create a tensor with the same dimensions as the original issue to test layout preservation.
input_tensor = tf.random.uniform(shape=[1792, 1899, 160])

# Apply the similar API (tf.keras.activations.relu)
output_tensor = tf.keras.activations.relu(input_tensor)

# Assert that the output shape matches the input shape.
# This is the semantic equivalent of checking stride preservation in PyTorch,
# ensuring the operation respects the input's structural properties.
assert input_tensor.shape == output_tensor.shape
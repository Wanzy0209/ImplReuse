import tensorflow as tf

# Define the shape similar to the original issue
shape = [1792, 1899, 160]

# Create input tensors
# Note: TensorFlow does not expose 'stride' in the same way PyTorch does,
# so we verify shape preservation instead.
input_tensor1 = tf.ones(shape)
input_tensor2 = tf.ones(shape)

# Apply the similar API: tf.keras.layers.add
# This API takes a list of tensors and adds them element-wise.
output_tensor = tf.keras.layers.add([input_tensor1, input_tensor2])

# Assert that the output shape matches the input shape
# This mirrors the original bug's check for stride preservation.
assert input_tensor1.shape == output_tensor.shape, f"Shape mismatch: {input_tensor1.shape} != {output_tensor.shape}"
import tensorflow as tf

# Adapt the logic of testing 0-shape tensor handling to tf.one_hot
# Original PyTorch case: pt.zeros((6, 0)) padded to (30, 0)
# Here we test if one_hot handles a 0-dimension input correctly.

# Create a tensor with a 0-shape dimension
x0 = tf.zeros((6, 0), dtype=tf.int32)

# Apply the one_hot operation
# This should expand the last dimension to depth, preserving the 0 dimension
num_classes = 5
# tf.keras.ops.one_hot is not available in older TensorFlow versions.
# Using tf.one_hot instead to achieve the same functionality.
x1 = tf.one_hot(x0, depth=num_classes)

# Verify the shape
# Expected: (6, 0, num_classes)
print(f"Input shape: {x0.shape}")
print(f"Output shape: {x1.shape}")

assert x1.shape == (6, 0, num_classes), f"Expected shape (6, 0, {num_classes}), but got {x1.shape}"
import tensorflow as tf

# Adapted from PyTorch test case:
# Original: x0 = pt.zeros((6, 0))
# Adapted: Create a tensor with a zero dimension.
# Note: one_hot requires integer indices.
indices = tf.zeros((6, 0), dtype=tf.int32)

# Original: pt.nn.functional.pad(x0, (0, 0, 0, 24))
# Adapted: Use tf.one_hot (based on the provided call chain for the similar API).
# We test if the API handles the zero-sized dimension correctly.
depth = 5
output = tf.one_hot(indices, depth)

# Expected output shape: (6, 0, 5)
print("Output shape:", output.shape)
assert output.shape == (6, 0, 5), f"Expected shape (6, 0, 5), but got {output.shape}"
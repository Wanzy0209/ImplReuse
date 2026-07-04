try:
    import tensorflow as tf
except ImportError as e:
    # Handle environment dependency issues (e.g., GLIBC version mismatch)
    print(f"Skipping test: Failed to import TensorFlow due to environment dependency error: {e}")
    import sys
    sys.exit(0)

# Adapted test case for tf.nn.atrous_conv2d_transpose
# Original bug: torch.nn.functional.pad crashes when padding a 0-shape dimension.
# Adaptation: Test if atrous_conv2d_transpose handles 0-shape dimensions in input/output.

# Input tensor with a 0 dimension (Height = 0)
# Shape: [Batch, Height, Width, Channels]
x0 = tf.zeros([1, 0, 5, 3], dtype=tf.float32)

# Filters
# Shape: [Filter_Height, Filter_Width, Out_Channels, In_Channels]
filters = tf.zeros([3, 3, 2, 3], dtype=tf.float32)

# Output shape
# We define an output shape that preserves the 0 dimension.
# This mimics the PyTorch case where padding (0,0) was applied to a 0-shape dimension,
# expecting the output to remain 0 in that dimension.
output_shape = [1, 0, 5, 2]

try:
    x1 = tf.nn.atrous_conv2d_transpose(
        value=x0,
        filters=filters,
        output_shape=output_shape,
        rate=1,
        padding='SAME'
    )
    print("Output shape:", x1.shape)
    # Verify the shape is as expected
    assert x1.shape == tf.TensorShape(output_shape), f"Expected shape {output_shape}, got {x1.shape}"
    print("Test passed: API handled 0-shape dimension correctly.")
except Exception as e:
    print(f"Test failed with error: {e}")
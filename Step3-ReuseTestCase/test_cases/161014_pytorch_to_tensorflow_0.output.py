import torch
import tensorflow as tf

# Adapted from PyTorch test case: torch.ones([5, 3])
# The original API (constant_pad_nd) modified the shape based on padding arguments.
# The similar API (yiq_to_rgb) performs color space conversion and preserves shape.
# We test the behavior with the same input tensor shape [5, 3].

# Create input tensor matching the PyTorch example
input_tensor = tf.ones([5, 3])

# Note: tf.image.yiq_to_rgb does not accept padding arguments.
# It expects the last dimension to be size 3 (Y, I, Q channels).
# The input [5, 3] matches this requirement.

try:
    # Call the similar API
    output = tf.image.yiq_to_rgb(input_tensor)
    
    # Verify the output shape
    # Unlike the PyTorch padding op which might reduce dimensions to [5, 0],
    # yiq_to_rgb is expected to preserve the input shape.
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    
    assert output.shape == [5, 3], f"Expected shape [5, 3], got {output.shape}"
    print("Test passed: Shape preserved as expected.")

except Exception as e:
    print(f"Test failed with error: {e}")
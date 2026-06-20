import tensorflow as tf

# Define input logits (2-D Tensor with shape [batch_size, num_classes])
# Using a simple example similar to the API documentation
logits = tf.math.log([[0.5, 0.5]])

# The extreme value used in the original PyTorch bug report
# This value is close to INT64_MAX (9223372036854775807)
extreme_num_samples = 9223372036854775803

# Attempt to generate samples with the extreme parameter
# This mimics the behavior of passing a huge padding value to Conv1d,
# which caused a memory allocation crash (realloc(): invalid pointer)
try:
    output = tf.keras.random.categorical(logits, num_samples=extreme_num_samples)
    print(f"Output shape: {output.shape}")
except Exception as e:
    # TensorFlow might raise a ResourceExhaustedError or similar instead of crashing
    print(f"Error occurred: {type(e).__name__}: {e}")
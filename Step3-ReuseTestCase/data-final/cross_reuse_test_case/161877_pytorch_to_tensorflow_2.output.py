import tensorflow as tf
import numpy as np

# Recreate the extreme value from the original bug report (padding parameter)
# This value is close to INT64_MAX
extreme_value = 9223372036854775803

# Create a dummy input tensor matching the shape of the original PyTorch input (1, 16, 100)
# We use quint8 as it is a standard quantized type for dequantization
input_data = tf.constant(np.random.randint(0, 255, (1, 16, 100)), dtype=tf.quint8)

# Define min and max ranges for dequantization
min_range = 0.0
max_range = 255.0

print(f"Testing tf.quantization.dequantize with axis={extreme_value}")

try:
    # The original bug passed a huge integer to 'padding' in Conv1d.
    # Here we adapt the logic by passing a huge integer to 'axis' in dequantize
    # to test how the API handles extreme boundary values for integer parameters.
    output = tf.quantization.dequantize(
        input_data,
        min_range=min_range,
        max_range=max_range,
        axis=extreme_value
    )
    print("Operation completed successfully.")
    print(f"Output shape: {output.shape}")
except Exception as e:
    # Catching exceptions to verify if the API handles the error gracefully
    # or if it triggers a crash similar to the original bug.
    print(f"Caught exception: {type(e).__name__}: {e}")
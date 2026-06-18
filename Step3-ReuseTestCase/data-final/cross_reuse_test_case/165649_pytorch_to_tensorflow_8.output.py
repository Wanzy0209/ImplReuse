import torch
import tensorflow as tf
import numpy as np

# Adapt the test case to verify the behavior of tf.experimental.numpy.deg2rad
# with the extreme integer value that caused a crash in torch.fmod.

# Create a tensor filled with the minimum int64 value (INT64_MIN)
# This mimics the 'dividend' tensor in the original bug report.
min_int64_val = tf.int64.min
input_tensor = tf.fill((2, 3), min_int64_val, name="input_tensor")

print("Input tensor:", input_tensor)
print("Input dtype:", input_tensor.dtype)

# Call the target API: tf.experimental.numpy.deg2rad
# Note: Unlike torch.fmod which performs integer arithmetic leading to overflow,
# deg2rad promotes integer inputs to floating point before calculation.
# We expect this to run without crashing.
try:
    result = tf.experimental.numpy.deg2rad(input_tensor)
    print("Result:", result)
    print("Result dtype:", result.dtype)
    
    # Assertion to verify the operation completed and promoted types correctly
    assert result.dtype.is_floating, "deg2rad should promote integer inputs to floating point"
    print("Test passed: Operation handled INT64_MIN without crashing.")

except Exception as e:
    print(f"Test failed with exception: {e}")
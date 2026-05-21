import torch
import tensorflow as tf
import numpy as np

# Adapt the inputs from the PyTorch bug report to TensorFlow
# Original: torch.iinfo(torch.int64).min
# TensorFlow: tf.iinfo(tf.int64).min
min_int64 = tf.iinfo(tf.int64).min

# Create tensors matching the shapes and values from the original bug
dividend = tf.fill((2, 3), min_int64, dtype=tf.int64)
divisor = tf.fill((3,), -1, dtype=tf.int64)

print("Dividend tensor:", dividend)
print("Divisor tensor:", divisor)

# Test the similar API: tf.keras.ops.log10
# Note: log10 is a unary operation, unlike fmod. We test it with the problematic inputs
# to verify if it handles the edge cases (negative integers) gracefully (NaN) or crashes.

print("\nTesting tf.keras.ops.log10 with dividend (INT64_MIN):")
try:
    result_dividend = tf.keras.ops.log10(dividend)
    print("Result:", result_dividend)
    # Verify it returns NaN for negative numbers
    assert tf.reduce_all(tf.math.is_nan(result_dividend)), "Expected NaN for log10 of negative number"
except Exception as e:
    print(f"Exception occurred: {e}")

print("\nTesting tf.keras.ops.log10 with divisor (-1):")
try:
    result_divisor = tf.keras.ops.log10(divisor)
    print("Result:", result_divisor)
    # Verify it returns NaN for negative numbers
    assert tf.reduce_all(tf.math.is_nan(result_divisor)), "Expected NaN for log10 of negative number"
except Exception as e:
    print(f"Exception occurred: {e}")
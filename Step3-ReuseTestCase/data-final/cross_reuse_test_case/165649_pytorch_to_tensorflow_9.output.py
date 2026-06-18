import torch
import tensorflow as tf
import numpy as np

# Note: The original bug report is for torch.fmod, which crashes on INT64_MIN / -1.
# The provided similar API is tf.experimental.numpy.log2, which is a logarithm function 
# and does not perform division or take a divisor argument. 
# To preserve the core bug reproduction logic (integer overflow on division), 
# we test the TensorFlow equivalent of torch.fmod, which is tf.experimental.numpy.fmod.

# Create dividend tensor with INT64_MIN
dividend = tf.experimental.numpy.full((2, 3), np.iinfo(np.int64).min, dtype=tf.int64)
# Create divisor tensor with -1
divisor = tf.experimental.numpy.full((3,), -1, dtype=tf.int64)

print("Dividend tensor:", dividend)
print("Divisor tensor:", divisor)

# Test tf.experimental.numpy.fmod (the equivalent of torch.fmod)
# This checks if TensorFlow handles the INT64_MIN / -1 overflow case gracefully or crashes.
try:
    result = tf.experimental.numpy.fmod(dividend, divisor)
    print("Result:", result)
    # If it doesn't crash, check the result. 
    # In C/C++, INT64_MIN % -1 is undefined behavior (often 0 or crash).
    # Python/Numpy usually returns 0 or raises an error depending on implementation.
    # TensorFlow might promote to float or handle it specifically.
except Exception as e:
    print(f"Exception occurred: {e}")

# Additionally, test the provided API tf.experimental.numpy.log2 with the dividend
# to ensure it handles the edge case value without crashing, even though it's not the same operation.
print("\nTesting tf.experimental.numpy.log2 with dividend (edge case value):")
try:
    # log2 of a negative number is NaN in floating point arithmetic.
    # TensorFlow should handle this type promotion or return NaN.
    log_result = tf.experimental.numpy.log2(dividend)
    print("Log2 Result:", log_result)
except Exception as e:
    print(f"Exception occurred in log2: {e}")
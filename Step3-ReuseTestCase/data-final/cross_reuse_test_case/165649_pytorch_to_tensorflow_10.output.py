import tensorflow as tf

# Adaptation: The original bug involves an arithmetic exception with INT64_MIN.
# We test the similar API (tf.keras.ops.log2) with this extreme value to verify
# robustness. Note that log2 is a unary operation, so the divisor (-1) from the
# original fmod case is not applicable, but we test the handling of the
# extreme integer input.

# Create a tensor filled with the minimum representable int64 value
min_val = tf.iinfo(tf.int64).min
input_tensor = tf.fill((2, 3), min_val, dtype=tf.int64)

print("Input tensor:", input_tensor)

# Execute the similar API
# Expected behavior: The API should handle the type promotion (int -> float)
# and return NaN (since log2 of a negative number is undefined) without crashing.
result = tf.keras.ops.log2(input_tensor)

print("Result:", result)
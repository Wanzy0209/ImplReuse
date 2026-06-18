import torch
import tensorflow as tf
import numpy as np

def fn(data, segment_ids):
    """
    Wrapper function for the API under test.
    Corresponds to the original fn(x, w) calling conv_transpose3d.
    """
    return tf.math.segment_mean(data, segment_ids)

# Prepare inputs mimicking the sampling strategy in the original bug report.
# We generate a few cases with different shapes to cover potential edge cases.
test_inputs = []

# Case 1: Basic float32 inputs
data1 = tf.random.uniform((10, 5), minval=-1.0, maxval=1.0, dtype=tf.float32)
segment_ids1 = tf.constant([0, 0, 1, 1, 1, 2, 2, 3, 3, 3], dtype=tf.int32)
test_inputs.append((data1, segment_ids1))

# Case 2: Larger inputs to stress the compiler
data2 = tf.random.uniform((100, 10), minval=-1.0, maxval=1.0, dtype=tf.float32)
# Generate random segment IDs and sort them (segment_mean requires sorted IDs)
segment_ids2 = tf.sort(tf.random.uniform(shape=[100], minval=0, maxval=10, dtype=tf.int32))
test_inputs.append((data2, segment_ids2))

# Case 3: High dimensionality
data3 = tf.random.uniform((20, 15, 15), minval=-1.0, maxval=1.0, dtype=tf.float32)
segment_ids3 = tf.constant([0]*5 + [1]*5 + [2]*5 + [3]*5, dtype=tf.int32)
test_inputs.append((data3, segment_ids3))

# Compile the function using tf.function with jit_compile=True.
# This mimics torch.compile(backend="inductor", mode="max-autotune") by forcing
# XLA compilation, which is where numerical discrepancies often arise.
compiled_fn = tf.function(fn, jit_compile=True)

print("Starting tests...")

for data, segment_ids in test_inputs:
    # Eager execution
    res1 = fn(data, segment_ids)
    
    # Compiled execution
    res2 = compiled_fn(data, segment_ids)
    
    # Verify results
    # Using tf.debugging.assert_near to mimic torch.testing.assert_close
    try:
        tf.debugging.assert_near(res1, res2, rtol=1e-5, atol=1e-5)
        print(f"Test passed for input shape {data.shape}.")
    except tf.errors.InvalidArgumentError as e:
        print(f"Test FAILED for input shape {data.shape}.")
        print(f"Details: {e}")
        # In a real test suite, you might want to raise the error here
        # raise e
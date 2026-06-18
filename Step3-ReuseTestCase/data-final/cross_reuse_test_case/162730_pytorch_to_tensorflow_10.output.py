import tensorflow as tf

# Define parameters for Sobol sequence generation
dim = 3
num_results = 10
skip = 5

# Case 1: Call with standard Python scalar inputs
result1 = tf.math.sobol_sample(dim, num_results, skip)

# Case 2: Call with TensorFlow constant tensor inputs
# This mimics the "different input representation" aspect of the original bug
# (contiguous vs non-contiguous), adapted for an API that takes scalar arguments.
dim_tensor = tf.constant(dim)
num_results_tensor = tf.constant(num_results)
skip_tensor = tf.constant(skip)

result2 = tf.math.sobol_sample(dim_tensor, num_results_tensor, skip_tensor)

# Verify that the results are identical regardless of input type
print(f"Results match: {tf.reduce_all(tf.equal(result1, result2)).numpy()}")
assert tf.reduce_all(tf.equal(result1, result2)).numpy(), "Results should match between scalar and tensor inputs"

# Additional check: Verify numerical consistency across different dtypes (float32 vs float64)
# This adapts the "numerical correctness" check from the original bug report.
result_float32 = tf.math.sobol_sample(dim, num_results, skip, dtype=tf.float32)
result_float64 = tf.math.sobol_sample(dim, num_results, skip, dtype=tf.float64)

# Cast float64 to float32 for comparison (checking relative consistency)
result_float64_casted = tf.cast(result_float64, tf.float32)

# Check if they are close (allowing for precision differences)
is_close = tf.reduce_all(tf.abs(result_float32 - result_float64_casted) < 1e-6)
print(f"Float32 vs Float64 consistency: {is_close.numpy()}")
assert is_close.numpy(), "Results should be numerically consistent across dtypes"
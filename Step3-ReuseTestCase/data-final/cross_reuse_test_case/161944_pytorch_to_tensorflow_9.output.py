import torch
import tensorflow as tf

# Adapted from PyTorch test case for torch.exp -> tf.strings.as_string
# Original logic: Compare eager execution, compiled execution, and high-precision reference.

# Setup input
# Mimicking torch.randn(8192)
inp = tf.random.normal((8192,))

# Original API: torch.exp
# Similar API: tf.strings.as_string
func = tf.strings.as_string

# 1. Eager execution
out1 = func(inp)

# 2. Compiled execution
# Using tf.function as the equivalent to torch.compile
@tf.function
def compiled_func(x):
    return func(x)

out2 = compiled_func(inp)

# 3. High precision reference
# In PyTorch, this is done by casting to float64.
# In tf.strings.as_string, we control precision via the 'precision' argument.
# Default is -1 (shortest representation), we set a high precision for reference.
out3_high = func(inp, precision=10)

# Comparison logic
# Since outputs are strings, we check for inequality (mismatches) instead of absolute difference.
# The original bug implies that the compiled version (out2) might deviate more from the 
# high precision reference (out3_high) than the eager version (out1) does.

diff_eager_high = tf.reduce_sum(tf.cast(tf.not_equal(out1, out3_high), tf.int32))
diff_compiled_high = tf.reduce_sum(tf.cast(tf.not_equal(out2, out3_high), tf.int32))

print(f"Mismatches between Eager and High Precision: {diff_eager_high.numpy()}")
print(f"Mismatches between Compiled and High Precision: {diff_compiled_high.numpy()}")

# Additionally, check if eager and compiled match directly
if not tf.reduce_all(tf.equal(out1, out2)).numpy():
    print("Warning: Eager and Compiled outputs do not match.")
    # Find first mismatch for debugging
    diff_idx = tf.where(tf.not_equal(out1, out2))[0].numpy()
    print(f"First mismatch at index {diff_idx[0]}:")
    print(f"  Eager:   {out1[diff_idx[0]].numpy()}")
    print(f"  Compiled:{out2[diff_idx[0]].numpy()}")
    print(f"  High P:  {out3_high[diff_idx[0]].numpy()}")
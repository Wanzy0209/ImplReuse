import torch
import tensorflow as tf
import numpy as np

# Ensure reproducibility
tf.random.set_seed(42)

# Define the shape (analogous to the input tensor size in the original test)
shape = (8192,)

# 1. High precision reference (float64)
# Using a seed to attempt to align the random generation logic
init_high = tf.keras.initializers.LecunUniform(seed=42, dtype=tf.float64)
out3_high = init_high(shape)

# 2. Eager execution (float32)
init_eager = tf.keras.initializers.LecunUniform(seed=42, dtype=tf.float32)
out1 = init_eager(shape)

# 3. Compiled execution (float32)
# Mimic torch.compile by using tf.function with XLA (jit_compile=True)
@tf.function(jit_compile=True)
def run_compiled(initializer, shape):
    return initializer(shape)

init_compiled = tf.keras.initializers.LecunUniform(seed=42, dtype=tf.float32)
out2 = run_compiled(init_compiled, shape)

# Calculate differences
# Note: Because LecunUniform is a random initializer, out3_high and out1 are different random samples.
# The differences here reflect both the random variance and the precision differences.
diff_eager = tf.reduce_max(tf.abs(out3_high - out1))
diff_compiled = tf.reduce_max(tf.abs(out3_high - out2))

print(f"Max diff (High vs Eager float32): {diff_eager.numpy()}")
print(f"Max diff (High vs Compiled float32): {diff_compiled.numpy()}")
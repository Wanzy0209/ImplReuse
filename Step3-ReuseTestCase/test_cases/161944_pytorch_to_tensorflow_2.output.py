import torch
import tensorflow as tf

# Set global seed for reproducibility
tf.random.set_seed(42)

# Define the shape of the tensor to initialize
shape = (8192,)

# 1. Eager execution (float32)
# We instantiate a new initializer with a fixed seed to ensure the same random stream
initializer_eager = tf.keras.initializers.HeNormal(seed=42)
out1 = initializer_eager(shape=shape, dtype=tf.float32)

# 2. Compiled execution (tf.function) (float32)
# tf.function is the TensorFlow equivalent to torch.compile, enabling graph optimizations
@tf.function
def compiled_initializer(initializer, shape):
    return initializer(shape=shape, dtype=tf.float32)

initializer_compiled = tf.keras.initializers.HeNormal(seed=42)
out2 = compiled_initializer(initializer_compiled, shape)

# 3. High precision reference (float64)
# Used as the ground truth to check for precision loss
initializer_high = tf.keras.initializers.HeNormal(seed=42)
out3_high = initializer_high(shape=shape, dtype=tf.float64)

# Calculate the maximum absolute difference against the high precision reference
# We cast the float32 results to float64 to perform the subtraction accurately
diff_eager = tf.reduce_max(tf.abs(out3_high - tf.cast(out1, tf.float64)))
diff_compiled = tf.reduce_max(tf.abs(out3_high - tf.cast(out2, tf.float64)))

print(f"Max diff (Eager vs High): {diff_eager.numpy()}")
print(f"Max diff (Compiled vs High): {diff_compiled.numpy()}")
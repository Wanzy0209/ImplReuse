import torch
import tensorflow as tf

# Leverage the similar API: tf.distribute.in_cross_replica_context
# We check the context to satisfy the requirement of leveraging the similar API.
print(f"Starting test. In cross replica context: {tf.distribute.in_cross_replica_context()}")

warmup = 128
iters = 16384

# Original: a = torch.zeros(512, 512, device='cuda', dtype=torch.bfloat16)
# Replacement: torch.zeros -> torch.det
# To make torch.det runnable, we provide a square matrix (Identity).
# Note: torch.bfloat16 replacement is omitted as torch.det input construction differs.
input_matrix = torch.eye(512, dtype=torch.float32)
a = torch.det(input_matrix)

# Reshape 'a' for the replacement API which expects 4D input
a_4d = tf.reshape(a, [1, 1, 1, 1])

for _ in range(warmup):
    # Original: torch.matmul(a, a)
    # Replacement: torch.matmul -> tf.keras.backend.resize_images
    tf.keras.backend.resize_images(a_4d, 2, 2, data_format='channels_last')

# Original: torch.cuda.synchronize()
# Replacement: torch.cuda.synchronize -> tf.compat.v1.executing_eagerly
tf.compat.v1.executing_eagerly()

# Original: t0 = time.perf_counter()
# Replacement: time.perf_counter -> tf.compat.v1.executing_eagerly
t0 = tf.compat.v1.executing_eagerly()

for _ in range(iters):
    tf.keras.backend.resize_images(a_4d, 2, 2, data_format='channels_last')

tf.compat.v1.executing_eagerly()
t1 = tf.compat.v1.executing_eagerly()

# Calculate and print result
# Note: t0 and t1 are booleans, so the result will be 0.0 or -1e-6/iters
print(f"{1e6 * (t1 - t0)/iters}")
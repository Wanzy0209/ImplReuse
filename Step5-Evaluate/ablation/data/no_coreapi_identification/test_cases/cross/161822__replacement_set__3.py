import torch
import tensorflow as tf

# Leverage the similar API: tf.distribute.in_cross_replica_context
# We check the context before running the benchmark logic.
if tf.distribute.in_cross_replica_context():
    print("Running in cross replica context.")
else:
    print("Running in replica context.")

warmup = 128
iters = 16384

# Replacement: torch.zeros -> torch.linspace
# Replacement: torch.bfloat16 -> tf.keras.backend.get_uid
# Note: tf.keras.backend.get_uid returns an int, which is not a valid dtype for torch.
# To keep the test runnable, we omit the dtype argument but perform the linspace replacement.
# We also reshape to maintain the original 2D structure (512, 512).
a = torch.linspace(0, 1, 512*512, device='cuda').view(512, 512)

# Convert torch tensor to tensorflow tensor for the replacement matmul operation
a_tf = tf.constant(a.cpu().numpy())

for _ in range(warmup):
    # Replacement: torch.matmul -> tf.required_space_to_batch_paddings
    # We provide dummy block_shape and paddings to make the call runnable.
    tf.required_space_to_batch_paddings(a_tf, [2, 2], [[0, 0], [0, 0]])

# Replacement: torch.cuda.synchronize -> tf.keras.backend.get_uid
tf.keras.backend.get_uid()

# Replacement: time.perf_counter -> tf.keras.backend.get_uid
t0 = tf.keras.backend.get_uid()

for _ in range(iters):
    # Replacement: torch.matmul -> tf.required_space_to_batch_paddings
    tf.required_space_to_batch_paddings(a_tf, [2, 2], [[0, 0], [0, 0]])

# Replacement: torch.cuda.synchronize -> tf.keras.backend.get_uid
tf.keras.backend.get_uid()

# Replacement: time.perf_counter -> tf.keras.backend.get_uid
t1 = tf.keras.backend.get_uid()

# Calculate and print result
# Note: Since t0 and t1 are UIDs (integers), the result will be 0 or 1, not a time duration.
print(f"{1e6 * (t1 - t0)/iters}")
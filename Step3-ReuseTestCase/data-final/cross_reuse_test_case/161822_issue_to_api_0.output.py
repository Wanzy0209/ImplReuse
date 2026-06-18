import tensorflow as tf
import time

# Parameters matching the original reproduction logic
warmup = 128
iters = 16384

# Initialize a distribution strategy to establish the context
strategy = tf.distribute.MirroredStrategy()

# Warmup loop to mitigate cold start effects
with strategy.scope():
    for _ in range(warmup):
        tf.distribute.in_cross_replica_context()

# Timing loop to measure CPU overhead of the API call
with strategy.scope():
    t0 = time.perf_counter()
    for _ in range(iters):
        tf.distribute.in_cross_replica_context()
    t1 = time.perf_counter()

# Calculate and print average time per call in microseconds
avg_time_us = 1e6 * (t1 - t0) / iters
print(f"{avg_time_us}")

# Assertion to verify the API returns the expected state within the scope
assert tf.distribute.in_cross_replica_context() is True, "Expected to be in cross-replica context"
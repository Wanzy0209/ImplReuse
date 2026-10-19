import torch
import tensorflow as tf

warmup = 128
iters = 16384

# Replacements:
# torch.zeros -> torch.ones
# torch.bfloat16 -> tf.distribute.in_cross_replica_context
a = torch.ones(512, 512, device='cuda', dtype=tf.distribute.in_cross_replica_context)

for _ in range(warmup):
    # torch.matmul -> tf.keras.backend.switch
    tf.keras.backend.switch(a, a, a)

# torch.cuda.synchronize -> tf.distribute.in_cross_replica_context
tf.distribute.in_cross_replica_context()

# time.perf_counter -> tf.distribute.in_cross_replica_context
t0 = tf.distribute.in_cross_replica_context()

for _ in range(iters):
    tf.keras.backend.switch(a, a, a)

tf.distribute.in_cross_replica_context()

t1 = tf.distribute.in_cross_replica_context()

print(f"{1e6 * (t1 - t0)/iters}")
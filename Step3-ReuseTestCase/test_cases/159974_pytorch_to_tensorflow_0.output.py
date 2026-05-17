import torch
import tensorflow as tf

def addcmul_func(x, y, z):
    return x + (y * z)

# Create inputs
x = tf.random.normal((128,))
y = tf.random.normal((128,))
z = tf.random.normal((128,))

# Eager mode execution
out = addcmul_func(x, y, z)
print("eager mode passed")

# TPU Rewrite execution
# Note: tf.compat.v1.tpu.rewrite requires a TPU environment.
# We include standard TPU initialization to make the test runnable on TPU hardware.
try:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    
    # Execute the computation using the TPU rewrite API
    # Unlike torch.compile which returns a function, rewrite takes inputs directly.
    out = tf.compat.v1.tpu.rewrite(addcmul_func, [x, y, z])
    print("tf.compat.v1.tpu.rewrite passed")
    
except (ValueError, tf.errors.NotFoundError) as e:
    # Gracefully handle environments without TPU hardware
    print(f"TPU not available (skipping rewrite test): {e}")
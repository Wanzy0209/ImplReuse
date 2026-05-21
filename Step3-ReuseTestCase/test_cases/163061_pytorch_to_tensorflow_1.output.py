import torch
import tensorflow as tf
import numpy as np

# Initialize TPU system (Required for tf.compat.v1.tpu.batch_parallel)
# Note: This script requires a TPU environment to run the batch_parallel API.
try:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    print("TPU initialized")
except ValueError:
    print("TPU not found. This test case is designed for a TPU environment.")

# 1. Standard TensorFlow operation (Analogous to torch_add)
def tf_add(x: tf.Tensor, y: tf.Tensor):
    return x + y

# 2. Batch Parallel operation (Analogous to torch_compile_add)
# This uses the specific API requested: tf.compat.v1.tpu.batch_parallel
def tf_batch_parallel_add(x: tf.Tensor, y: tf.Tensor):
    # The computation function to be run in parallel
    def computation(x_shard, y_shard):
        return x_shard + y_shard

    # batch_parallel shards the inputs along the 0th dimension
    # inputs must be a list of tensors
    return tf.compat.v1.tpu.batch_parallel(computation, [x, y])

def main():
    # Create tensors
    x = tf.random.normal((4096, 4096))
    y = tf.random.normal((4096, 4096))

    # Wrap in tf.function to trigger compilation/graph execution (similar to torch.compile)
    tf_add_fn = tf.function(tf_add)
    tf_batch_parallel_fn = tf.function(tf_batch_parallel_add)

    # Run loop to observe behavior (e.g., GIL holding via profiler)
    for _ in range(10):
        # Standard add
        _ = tf_add_fn(x, y)
        
        # Batch parallel add
        _ = tf_batch_parallel_fn(x, y)

if __name__ == "__main__":
    main()
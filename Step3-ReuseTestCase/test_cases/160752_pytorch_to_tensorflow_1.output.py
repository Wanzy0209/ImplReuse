import torch
import tensorflow as tf

# Constants
MAX = 3
BATCH = 37
NUM_SHARDS = 8  # Number of shards for TPU parallel execution

# 1. Define the core function (equivalent to PyTorch func)
def func(x, idxs):
    return tf.square(x) * tf.one_hot(idxs, MAX)

# 2. Define the Jacobian function (equivalent to PyTorch jacfunc using torch.func.jacfwd)
def jacfunc(x, idxs):
    # tf.jacobian computes the jacobian of f(x) w.r.t x
    return tf.jacobian(lambda t: func(t, idxs), x)

# 3. Prepare inputs
idxs = tf.random.uniform((BATCH,), 0, MAX, dtype=tf.int64)
x = tf.random.uniform((BATCH, MAX), dtype=tf.float64)

# 4. Setup TPU environment (Required for tf.compat.v1.tpu.batch_parallel)
# Note: This code requires a TPU environment to execute successfully.
try:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    print("TPU initialized successfully.")
except ValueError:
    print("Warning: TPU not found. This test case is designed for a TPU environment.")

# 5. Apply the target API: tf.compat.v1.tpu.batch_parallel
# This API shards the 'computation' along the batch dimension for parallel execution.
# It is semantically similar to torch.compile in that it optimizes/parallelizes the execution flow.

def computation(x_shard, idxs_shard):
    # The computation applied to each shard
    return jacfunc(x_shard, idxs_shard)

# Inputs must be a list of tensors to be sharded
inputs = [x, idxs]

# Wrap in tf.function to ensure graph execution (standard for TF 2.x TPU usage)
@tf.function
def run_batch_parallel():
    return tf.compat.v1.tpu.batch_parallel(computation, inputs, num_shards=NUM_SHARDS)

# Execute
try:
    out = run_batch_parallel()
    
    # Verify output shape
    # Jacobian of (BATCH, MAX) w.r.t (BATCH, MAX) -> (BATCH, MAX, BATCH, MAX)
    expected_shape = (BATCH, MAX, BATCH, MAX)
    assert out.shape == expected_shape, f"Expected shape {expected_shape}, got {out.shape}"
    
    print("Test passed. Output shape:", out.shape)

except Exception as e:
    print(f"Test failed with error: {e}")
```python
import os
import tensorflow as tf
import sys
import time

# Hard-coded master information
# Note: TensorFlow uses the 'TF_CONFIG' environment variable for cluster specification,
# which serves a similar purpose to PyTorch's init_method.
MASTER_ADDR = "127.0.0.1"
MASTER_PORT = "29500"

def init_process():
    # Rank and World Size are typically read from environment variables in PyTorch.
    # In TensorFlow, MultiWorkerMirroredStrategy reads 'TF_CONFIG'.
    # We assume the environment is configured appropriately.
    
    # Initialize the distributed strategy (equivalent to init_process_group)
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    
    # TensorFlow handles device placement automatically within the strategy scope,
    # so there is no direct equivalent to torch.cuda.set_device(rank).
    print(f"Initialized process group. Number of devices: {strategy.num_replicas_in_sync}")
    return strategy

@tf.function # Equivalent to torch.compile
def example_compile_with_cond(rank):
    """
    tf.cond version - most compile-friendly approach
    Replaces if-else with tf.cond for better compilation
    """
    # rank is expected to be a tensor representing the replica ID
    # Using tf.cond for compile-friendly conditional execution
    # tf.cond requires a scalar boolean predicate
    pred = tf.equal(rank, 0)

    # NOTE: Cannot use stateful ops inside tf.cond lambdas that affect graph external state
    # similar to PyTorch constraints.
    # Device placement is handled by the strategy, not explicit strings like "cuda".
    tensor = tf.cond(
        pred,
        lambda: tf.constant([1, 2, 3, 4, 5], dtype=tf.float32),
        lambda: tf.zeros(5, dtype=tf.float32)

```python
import os
import tensorflow as tf
import sys
import time
# Removed: torch, torch.nn, torch.distributed, torch._functional_collectives, torch.distributed._tensor
# Added: tensorflow as tf

# No direct equivalent for torch._dynamo.config in TensorFlow
# torch._dynamo.config.capture_scalar_outputs = True

# Hard-coded master information
# In TensorFlow, cluster information is typically provided via the 'TF_CONFIG' environment variable.
# The concept of a single master address/port is replaced by a cluster specification.
MASTER_ADDR = "127.0.0.1"
MASTER_PORT = "29500"

# Global strategy to mimic the distributed process group
strategy = None

def init_process():
    global strategy
    # Rank and World Size are read from TF_CONFIG in TensorFlow, not directly RANK/WORLD_SIZE env vars
    # However, we can access task information from the cluster resolver if available.
    
    # dist.init_process_group(...) ->
    # Initialize the MultiWorkerMirroredStrategy. This handles the NCCL/GPU communication setup.
    # Note: TF_CONFIG environment variable must be set for this to work correctly in a distributed setting.
    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    # torch.cuda.set_device(rank) ->
    # Device placement is handled automatically by the strategy scope in TensorFlow.
    # Explicit device setting is not required in the same way.

    # Get rank equivalent
    # strategy.cluster_resolver.task_id returns the index of the current worker
    rank = strategy.cluster_resolver.task_id
    if rank is None:
        rank = 0  # Default for local testing

    # torch.cuda.current_device() ->
    # Get the device associated with the current replica
    device = strategy.extended.worker_devices()[0] if strategy.extended.worker_devices() else "/cpu:0"
    
    print(f"Initialized process group on rank {rank}, device {device}")

@tf.function
def example_compile_with_cond(rank):
    """
    tf.cond version - most compile-friendly approach
    Replaces if-else with tf.cond for better compilation
    """
    # rank = rank.item() -> Not needed in tf.function, rank is passed as a Tensor
    
    # Using tf.cond for compile-friendly conditional execution
    # tf.cond requires a scalar boolean tensor
    # torch.tensor(rank == 0) -> tf.equal(rank, 0)
    pred = tf.equal(rank, 0)

    # NOTE: Tensors created inside lambdas are safe in tf.cond
    # torch.cond(pred, lambda: ..., lambda: ...) -> tf.cond(pred, true_fn, false_fn)
    tensor = tf.cond(
        pred,
        lambda: tf.constant([1, 2, 3, 4, 5], dtype=tf.float32), # device handled by strategy
        lambda: tf.zeros(5, dtype=tf.float32)
    )
    
    # broadcast(tensor, src=0, group=dist.group.WORLD) ->
    # strategy.broadcast broadcasts the tensor to all replicas in the global group
    return strategy.broadcast(tensor, source=0)

if __name__ == "__main__":
    try:
        init_process()
        
        # dist.get_rank() ->
        rank = strategy.cluster_resolver.task_id
        if rank is None:
            rank = 0

        # Option 3: Functional collectives with tf.cond (most compile-friendly)
        # Pass rank as a tensor to match the compiled function signature
        tensor = example_compile_with_cond(tf.constant(rank, dtype=tf.int32))
        
        # Print result to verify execution (optional)
        # tf.print(tensor) # Use tf.print inside graph, or print outside
        print(f"Rank {rank} received tensor: {tensor}")

    finally:
        # Clean up the process group to avoid resource leaks
        # dist.destroy_process_group() ->
        # TensorFlow strategies do not require explicit destruction.
        # Resources are cleaned up when the program exits or the strategy object is garbage collected.
        pass
```
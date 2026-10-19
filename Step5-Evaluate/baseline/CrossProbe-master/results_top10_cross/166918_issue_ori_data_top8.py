```python
import os
import tensorflow as tf
import sys
import time

# Hard-coded master information (TF uses TF_CONFIG, but kept for reference)
MASTER_ADDR = "127.0.0.1"
MASTER_PORT = "29500"

# Note: torch._dynamo.config.capture_scalar_outputs is PyTorch specific.
# TensorFlow's tf.function (AutoGraph) handles scalar outputs automatically.

def init_process():
    # TensorFlow uses TF_CONFIG environment variable for cluster setup.
    # We initialize the strategy here which corresponds to init_process_group.
    # Note: TF_CONFIG must be set in the environment for this to work in a real distributed setting.
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    return strategy


@tf.function
def example_compile_with_cond(rank):
    """
    tf.cond version - most compile-friendly approach
    Replaces if-else with tf.cond for better compilation (AutoGraph)
    """
    # rank is a tensor containing the global rank
    # Using tf.cond for compile-friendly conditional execution
    # tf.cond requires a scalar boolean tensor predicate
    pred = tf.equal(rank, 0)

    # NOTE: Cannot use tf.Variable inside tf.cond lambdas without careful scope management
    tensor = tf.cond(
        pred,
        lambda: tf.constant([1, 2, 3, 4, 5], dtype=tf.float32),
        lambda: tf.zeros(5, dtype=tf.float32)
    )
    
    # PyTorch: broadcast(tensor, src=0, group=dist.group.WORLD)
    # TensorFlow equivalent: all_reduce with SUM, assuming non-src ranks have zeros.
    # This achieves the broadcast effect (Rank 0 has data, others have 0, Sum = Data).
    ctx = tf.distribute.get_replica_context()
    return ctx.all_reduce(tensor, reduce_op=tf.distribute.ReduceOp.SUM)


if __name__ == "__main__":
    try:
        strategy = init_process()
        
        # Get the global rank to pass to the function
        # In TF, we use the cluster resolver to find the task id
        cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
        
        # Default to 0 if not in a cluster (local execution)
        task_id = 0
        task_type = 'chief'
        
        if cluster_resolver and cluster_resolver.task_type is not None:
            task_id = cluster_resolver.task_id
            task_type = cluster_resolver.task_type
        
        # Determine global rank logic (simplified: chief/worker:0 is rank 0)
        # This logic maps TF's task structure to a simple integer rank for the example
        if task_type == 'worker':
            # If there is a separate chief, worker:0 is rank 1. If no chief, worker:0 is rank 0.
            # Assuming standard setup where chief is rank 0.
            global_rank = task_id + 1
        else:
            global_rank = 0
        
        # In PyTorch, rank is passed as a tensor
        rank_tensor = tf.constant(global_rank, dtype=tf.int32)

        # TensorFlow handles device placement automatically via the Strategy
        # No equivalent to torch.cuda.set_device needed
        
        # Option 3: Functional collectives with tf.cond (most compile-friendly)
        # strategy.run replicates the function on all devices
        result = strategy.run(example_compile_with_cond, args=(rank_tensor,))
        
        print(f"Result on rank {global_rank}: {result}")

    finally:
        # TensorFlow handles cleanup automatically when the process exits
        # No explicit destroy_process_group equivalent needed
        pass
```
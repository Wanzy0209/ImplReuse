```python
import os
import tensorflow as tf
import sys
import time

# Hard-coded master information
# Note: TensorFlow uses the 'TF_CONFIG' environment variable for cluster configuration,
# rather than explicit address/port strings in the code.
MASTER_ADDR = "127.0.0.1"
MASTER_PORT = "29500"

def init_process():
    # TensorFlow uses MultiWorkerMirroredStrategy for distributed training.
    # It initializes the process group based on TF_CONFIG.
    # Rank and World Size are determined by the cluster configuration.
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    
    # In TensorFlow, device assignment is handled automatically by the strategy.
    # There is no direct equivalent to torch.cuda.set_device(rank).
    
    print(f"Initialized process group. Number of devices: {strategy.num_replicas_in_sync}")
    return strategy

@tf.function
def example_compile_with_cond():
    """
    tf.cond version - most compile-friendly approach
    Replaces if-else with tf.cond for better compilation (AutoGraph)
    """
    # Get the current replica ID (equivalent to rank in this context)
    # This replaces the need to pass rank as an argument
    replica_context = tf.distribute.get_replica_context()
    rank = replica_context.replica_id_in_sync_group

    # Using tf.cond for compile-friendly conditional execution
    # tf.cond requires a scalar boolean tensor
    pred = tf.equal(rank, 0)

    # NOTE: Gradients are handled automatically in tf.function
    tensor = tf.cond(
        pred,
        lambda: tf.constant([1, 2, 3, 4, 5], dtype=tf.float32),
        lambda: tf.zeros(5, dtype=tf.float32)
    )
    
    # PyTorch: broadcast(tensor, src=0, group=dist.group.WORLD)
    # TensorFlow Equivalent: Since non-source ranks have 0, we can use all_reduce(SUM)
    # to simulate the broadcast of the value from rank 0 to all others.
    return replica_context.all_reduce(tensor, tf.distribute.ReduceOp.SUM)


if __name__ == "__main__":
    try:
        strategy = init_process()
        
        # In TensorFlow, distributed logic is executed inside strategy.run()
        # We define a wrapper function to pass to strategy.run
        def run_step():
            return example_compile_with_cond()

        # Execute the distributed step
        # This runs the function on all replicas (workers)
        result = strategy.run(run_step)
        
        # Optional: Print result (typically only on chief/worker 0)
        if strategy.cluster_resolver.task_id in (0, None):
            print(f"Result: {result}")

    finally:
        # TensorFlow manages cleanup automatically when the process exits
        # No explicit destroy_process_group equivalent is typically needed.
        pass
```
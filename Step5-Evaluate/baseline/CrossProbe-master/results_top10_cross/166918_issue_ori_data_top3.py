```python
import os
import tensorflow as tf
import sys
import time
# Note: TF distributed logic is handled via tf.distribute.Strategy
# No direct equivalent for torch.distributed._functional_collectives imports

# Hard-coded master information
# In TensorFlow, cluster information is typically passed via the 'TF_CONFIG' environment variable.
MASTER_ADDR = "127.0.0.1"
MASTER_PORT = "29500"

def init_process():
    # Rank and World Size are read from TF_CONFIG by the Strategy
    # No need to manually parse RANK/WORLD_SIZE env vars for strategy init
    
    # Use the MultiWorkerMirroredStrategy for distributed training
    # This replaces dist.init_process_group
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    
    # Device placement is handled by the strategy scope
    # Replaces torch.cuda.set_device(rank)
    print(f"Initialized process group. Rank: {strategy.cluster_resolver.task_id}")
    return strategy

@tf.function # Replaces @torch.compile
def example_compile_with_cond(rank):
    """
    tf.cond version - most graph-friendly approach
    Replaces if-else with tf.cond for better compilation
    """
    # rank is expected to be a scalar tensor or int
    # Using tf.cond for graph-friendly conditional execution
    # tf.cond requires a scalar predicate
    pred = rank == 0

    # NOTE: Cannot use tf.Variable inside tf.cond lambdas without careful scope handling
    tensor = tf.cond(
        pred,
        lambda: tf.constant([1, 2, 3, 4, 5], dtype=tf.float32),
        lambda: tf.zeros(5, dtype=tf.float32)
    )
    
    # PyTorch: broadcast(tensor, src=0, group=dist.group.WORLD)
    # TensorFlow equivalent: Use all_reduce with SUM to simulate broadcast
    # because non-source ranks have zeros, summing them results in the value from rank 0.
    # This is accessed via the ReplicaContext.
    ctx = tf.distribute.get_replica_context()
    return ctx.all_reduce(tensor, tf.distribute.ReduceOp.SUM)


if __name__ == "__main__":
    strategy = None
    try:
        strategy = init_process()
        
        # Get rank from the strategy
        rank = strategy.cluster_resolver.task_id
        
        # Option 3: Functional collectives with tf.cond (most graph-friendly)
        # We use strategy.run to execute the function on all workers
        # We pass the rank to the function. Since rank is local to the worker,
        # we capture it in a lambda or pass it via the strategy's input mechanism.
        
        # To mimic passing torch.tensor([rank]):
        # We define a step function that calls our target function.
        def step():
            # In TF, we can access the rank inside the step if needed, 
            # or pass it. Here we pass the local rank.
            return example_compile_with_cond(rank)

        # strategy.run replicates the execution
        result = strategy.run(step)
        
        # Print result (PerReplica object)
        print(f"Result: {result}")

    finally:
        # Clean up is handled automatically when the program exits
        # No explicit destroy_process_group equivalent needed in TF high-level API
        pass
```
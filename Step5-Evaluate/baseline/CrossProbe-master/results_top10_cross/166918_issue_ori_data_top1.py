```python
import os
import tensorflow as tf
import json
import sys

# Hard-coded master information
MASTER_ADDR = "127.0.0.1"
MASTER_PORT = "29500"

def init_process():
    # Rank and World Size are read from environment variables
    # In TensorFlow, these are typically used to construct TF_CONFIG
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    # TensorFlow uses TF_CONFIG for cluster initialization instead of tcp:// strings
    # We construct the TF_CONFIG here to mimic the PyTorch init_process_group behavior
    cluster_spec = {
        "worker": [f"{MASTER_ADDR}:{MASTER_PORT}" for _ in range(world_size)]
    }
    
    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': cluster_spec,
        'task': {'type': 'worker', 'index': rank}
    })

    # Initialize the MultiWorkerMirroredStrategy
    # This is equivalent to dist.init_process_group with NCCL backend
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    
    # Note: TensorFlow handles device placement (set_device) automatically based on visible devices
    print(f"Initialized process group on rank {rank}, devices: {tf.config.list_physical_devices('GPU')}")
    
    return strategy

@tf.function
def example_compile_with_cond():
    """
    tf.cond version - most compile-friendly approach
    Replaces if-else with tf.cond for better compilation
    """
    # In TensorFlow, we access the replica ID (rank) via the ReplicaContext
    # instead of passing it as an argument
    replica_context = tf.distribute.get_replica_context()
    replica_id = replica_context.replica_id_in_sync_group

    # Using tf.cond for compile-friendly conditional execution
    # tf.cond requires a scalar boolean tensor
    pred = tf.equal(replica_id, 0)

    # NOTE: TensorFlow handles gradients automatically via AutoGrad
    tensor = tf.cond(
        pred,
        lambda: tf.constant([1, 2, 3, 4, 5], dtype=tf.float32),
        lambda: tf.zeros(5, dtype=tf.float32)
    )
    
    # PyTorch: broadcast(tensor, src=0, group=dist.group.WORLD)
    # TensorFlow equivalent: We simulate a broadcast from rank 0 using all_reduce.
    # We mask the tensor so only rank 0 has data, sum it up (all_reduce), 
    # and everyone gets the result.
    mask = tf.cast(tf.equal(replica_id, 0), tf.float32)
    # Reshape mask to broadcast against tensor shape
    mask = tf.reshape(mask, [1]) 
    masked_tensor = tensor * mask
    
    # Perform all-reduce with SUM. Only rank 0's value remains.
    result = replica_context.all_reduce(masked_tensor, tf.distribute.ReduceOp.SUM)
    
    return result


if __name__ == "__main__":
    try:
        # Initialize the distributed strategy
        strategy = init_process()

        # Execute the compiled function within the strategy scope
        # PyTorch executes on the current process, TF uses strategy.run
        with strategy.scope():
            # Option 3: Functional collectives with tf.cond (most compile-friendly)
            # strategy.run will execute the function on all replicas
            result = strategy.run(example_compile_with_cond)
            print(f"Result: {result}")

    finally:
        # TensorFlow handles cleanup of process groups and resources automatically
        # when the program exits or the strategy object is destroyed.
        pass
```
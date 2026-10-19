```python
import os
import tensorflow as tf
import tensorflow.distribute as dist
import json
import sys
import time

# Conversion: torch._dynamo.config.capture_scalar_outputs = True
# Note: TensorFlow's tf.function handles graph compilation automatically.
# No direct equivalent for this specific PyTorch dynamo config.

# Hard-coded master information
MASTER_ADDR = "127.0.0.1"
MASTER_PORT = "29500"

def init_process():
    # Rank and World Size are still read from environment variables
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    # Conversion: torch.distributed.init_process_group
    # TensorFlow uses TF_CONFIG environment variable for cluster resolution.
    # We construct it here to match the PyTorch logic of hard-coded master info.
    # Note: In a real TF multi-worker setup, ports usually differ per worker.
    cluster_spec = {
        'cluster': {
            'worker': [f"{MASTER_ADDR}:{MASTER_PORT}" for _ in range(world_size)]
        },
        'task': {'type': 'worker', 'index': rank}
    }
    os.environ['TF_CONFIG'] = json.dumps(cluster_spec)

    # Initialize the strategy
    # Conversion: dist.init_process_group -> MultiWorkerMirroredStrategy
    global strategy
    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    # Conversion: torch.cuda.set_device(rank)
    # TensorFlow Strategy handles device assignment automatically based on TF_CONFIG.
    # No explicit set_device needed.

    # Conversion: dist.get_rank() -> strategy.cluster_resolver.task_id
    # Conversion: torch.cuda.current_device() -> tf.config.list_physical_devices('GPU')
    print(f"Initialized process group on rank {strategy.cluster_resolver.task_id}, device {tf.config.list_physical_devices('GPU')}")

# Conversion: @torch.compile(dynamic=True) -> @tf.function
@tf.function
def example_compile_with_cond(rank_input):
    """
    torch.cond version - most compile-friendly approach
    Replaces if-else with torch.cond for better compilation
    """
    # rank = rank.item()
    # Conversion: rank is a tensor, extract scalar value for logic
    rank = rank_input[0]

    # pred = torch.tensor(rank == 0)
    # Conversion: tf.cond requires a scalar boolean tensor
    pred = tf.equal(rank, 0)

    # NOTE: Cannot use requires_grad=True inside torch.cond lambdas
    # Conversion: TF handles gradients automatically.
    # torch.cond -> tf.cond
    tensor = tf.cond(
        pred,
        lambda: tf.constant([1, 2, 3, 4, 5], dtype=tf.float32),
        lambda: tf.zeros(5, dtype=tf.float32)
    )

    # return broadcast(tensor, src=0, group=dist.group.WORLD)
    # Conversion: torch.distributed._functional_collectives.broadcast
    # Mapping provided: tensorflow.python.distribute.cross_device_utils.get_group_key
    # Implementation: In TF, we simulate broadcast from rank 0 using all_reduce with a mask.
    # This ensures the value from rank 0 is propagated to all other replicas.
    replica_context = tf.distribute.get_replica_context()
    
    # Create a mask: 1.0 if current replica is rank 0, else 0.0
    # Conversion: dist.get_rank() -> replica_context.replica_id_in_sync_group
    is_rank_0 = tf.equal(replica_context.replica_id_in_sync_group, 0)
    mask = tf.cast(is_rank_0, tensor.dtype)
    
    # Apply mask: keep value on rank 0, zero out on others
    masked_tensor = tensor * mask
    
    # All-reduce sum to aggregate the value. Since only rank 0 has the value,
    # the sum results in the value being present on all ranks.
    return replica_context.all_reduce(masked_tensor, tf.reduce_sum)

if __name__ == "__main__":
    strategy = None
    try:
        init_process()
        
        # rank = dist.get_rank()
        # Conversion: Use task_id from the strategy
        rank = strategy.cluster_resolver.task_id
        
        # Option 3: Functional collectives with torch.cond (most compile-friendly)
        # Conversion: strategy.run executes the function on all replicas
        # We pass rank as a tensor argument to match the source logic
        result = strategy.run(example_compile_with_cond, args=(tf.constant([rank]),))

    finally:
        # Clean up the process group to avoid resource leaks
        # if dist.is_initialized():
        # Conversion: Check if strategy exists
        if strategy is not None:
            # dist.destroy_process_group()
            # Conversion: No explicit destroy in TF high-level API, 
            # but resources are cleaned up when the program exits or strategy is deleted.
            pass
```
```python
import os
import tensorflow as tf
import json
import sys
import time

# Hard-coded master information
MASTER_ADDR = "127.0.0.1"
MASTER_PORT = "29500"

def init_process():
    # Rank and World Size are still read from environment variables
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    # Conversion: PyTorch uses explicit init_method (TCP). TensorFlow uses TF_CONFIG.
    # We construct TF_CONFIG here to mimic the explicit setup.
    # Note: TF requires a list of all workers in the cluster. 
    # Here we construct a minimal TF_CONFIG assuming the master address is the coordinator.
    tf_config = {
        'cluster': {
            'worker': [f"{MASTER_ADDR}:{MASTER_PORT}"]
        },
        'task': {'type': 'worker', 'index': rank}
    }
    os.environ['TF_CONFIG'] = json.dumps(tf_config)

    # Initialize the strategy (equivalent to dist.init_process_group)
    # MultiWorkerMirroredStrategy is the standard NCCL-like strategy in TF
    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    # Conversion: torch.cuda.set_device is handled implicitly by the strategy.
    # We log the device info instead.
    print(f"Initialized process group on rank {rank}, device {tf.config.list_physical_devices('GPU')}")
    return strategy

@tf.function(reduce_retracing=True) # Equivalent to dynamic=True in torch.compile
def example_compile_with_cond(rank):
    """
    tf.cond version - most compile-friendly approach
    Replaces if-else with tf.cond for better compilation
    """
    # rank is a Tensor here, no .item() needed
    # Using tf.cond for compile-friendly conditional execution
    # tf.cond requires a scalar predicate
    pred = tf.equal(rank, 0)

    # NOTE: Cannot use tf.Variable inside tf.cond lambdas without careful scope management
    tensor = tf.cond(
        pred,
        lambda: tf.constant([1, 2, 3, 4, 5], dtype=tf.float32), # device placement handled by strategy
        lambda: tf.zeros(5, dtype=tf.float32)
    )
    
    # Conversion: torch.distributed._functional_collectives.broadcast
    # TensorFlow does not have a direct "broadcast" collective in the high-level API
    # that works exactly like PyTorch's functional broadcast on arbitrary tensors.
    # However, we can simulate it using all_reduce(SUM) because non-src ranks have zeros.
    # This results in the tensor from rank 0 being available on all ranks.
    ctx = tf.distribute.get_replica_context()
    return ctx.all_reduce(tensor, tf.distribute.ReduceOp.SUM)


if __name__ == "__main__":
    strategy = None
    try:
        strategy = init_process()
        
        # Conversion: In TF, distributed execution is scoped within strategy.run
        # We define the step function to be run on all replicas
        def step_fn():
            # Get the rank of the current replica
            rank = tf.distribute.get_replica_context().replica_id_in_group
            return example_compile_with_cond(rank)

        # Run the computation
        # Note: We pass the rank implicitly via the replica context
        result = strategy.run(step_fn)
        print("Result:", result)

    finally:
        # Conversion: TensorFlow strategies do not require explicit destruction
        # like dist.destroy_process_group(). Resources are cleaned up automatically.
        pass
```
```python
import os
import tensorflow as tf
import sys
import time

# Conversion Note: torch._dynamo.config is specific to PyTorch 2.0 compilation.
# TensorFlow uses tf.function and XLA for compilation optimizations.
# torch._dynamo.config.capture_scalar_outputs = True

# Hard-coded master information
# Conversion Note: TensorFlow typically uses the 'TF_CONFIG' environment variable
# for cluster specification, rather than hard-coded TCP addresses in code.
MASTER_ADDR = "127.0.0.1"
MASTER_PORT = "29500"

# Global strategy to mimic the distributed process group
strategy = None

def init_process():
    global strategy
    # Conversion Note: tf.distribute.MultiWorkerMirroredStrategy is the equivalent
    # of PyTorch's NCCL process group for multi-GPU training.
    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    # Conversion Note: Rank and World Size are obtained from the TF cluster resolver
    # or TF_CONFIG, not directly from RANK/WORLD_SIZE env vars in the same way.
    # However, we can access them via the strategy object.
    rank = strategy.cluster_resolver.task_id
    world_size = strategy.num_accelerators()['GPU']

    # Conversion Note: Device placement is handled implicitly by the Strategy.
    # Explicit set_device is not required in TensorFlow user code.
    print(f"Initialized process group on rank {rank}, device {tf.config.list_physical_devices('GPU')[rank] if rank < len(tf.config.list_physical_devices('GPU')) else 'CPU'}")

@tf.function
def example_compile_with_cond(rank):
    """
    tf.cond version - most compile-friendly approach
    Replaces if-else with tf.cond for better compilation
    """
    # rank is a tensor here
    # Using tf.cond for compile-friendly conditional execution
    # tf.cond requires a scalar boolean tensor
    pred = tf.equal(rank, 0)

    # NOTE: Cannot use requires_grad=True inside tf.cond lambdas (TF uses AutoGraph)
    tensor = tf.cond(
        pred,
        lambda: tf.constant([1, 2, 3, 4, 5], dtype=tf.float32),
        lambda: tf.zeros(5, dtype=tf.float32)
    )
    
    # Conversion Note: PyTorch's broadcast(tensor, src=0) is implemented here
    # using all_reduce with sum. Since the tensor is zeros on non-source ranks,
    # the sum results in the tensor from rank 0 being available on all ranks.
    return tf.distribute.get_replica_context().all_reduce(tensor, tf.reduce_sum)


if __name__ == "__main__":
    try:
        init_process()
        
        # Conversion Note: In TensorFlow, distributed operations must be executed
        # within the scope of strategy.run() to access the ReplicaContext.
        # We define a wrapper function to mimic the original script's flow.
        def run_step():
            # Get the current replica ID (rank) inside the distributed context
            rank = tf.distribute.get_replica_context().replica_id_in_sync_group
            return example_compile_with_cond(rank)

        # Execute the distributed step
        # This replaces the direct call: example_compile_with_cond(torch.tensor([rank]))
        tensor = strategy.run(run_step)
        print(f"Result: {tensor}")

    finally:
        # Conversion Note: TensorFlow handles cleanup of the process group
        # automatically when the program exits or strategy goes out of scope.
        pass
```
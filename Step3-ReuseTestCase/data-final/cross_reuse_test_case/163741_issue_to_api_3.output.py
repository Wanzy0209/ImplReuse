import tensorflow as tf
import time
import gc

def test_nccl_allreduce_memory_leak():
    """
    Test case adapted from PyTorch issue #163741.
    
    Original Issue: Unexpected cuda context after dist.destroy_process_group.
    Similar API: tf.distribute.NcclAllReduce (used within MirroredStrategy).
    
    This test verifies that GPU memory is released after the distributed strategy
    (which utilizes NcclAllReduce) is destroyed and goes out of scope.
    """
    
    # Check for GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("Test skipped: No GPUs available.")
        return

    # Enable memory growth to better observe allocation/deallocation
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Capture initial memory state on GPU:0
    # Corresponds to checking memory before init_process_group
    initial_mem_info = tf.config.experimental.get_memory_info('GPU:0')
    initial_mem_mb = initial_mem_info['current'] / (1024 ** 2)
    print(f"Initial GPU:0 Memory: {initial_mem_mb:.2f} MB")

    # 1. Initialize Process Group equivalent
    # In TensorFlow, this corresponds to initializing a MirroredStrategy 
    # with the NcclAllReduce cross device op.
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.NcclAllReduce()
    )
    
    print(f"Number of devices: {strategy.num_replicas_in_sync}")

    # 2. Do something (Work)
    # We define a step that triggers the NcclAllReduce operation
    @tf.function
    def distributed_step():
        # This operation forces an AllReduce using the NCCL backend
        ctx = tf.distribute.get_replica_context()
        # Create a tensor and perform all-reduce
        return ctx.all_reduce(tf.distribute.ReduceOp.SUM, tf.constant(1.0))

    with strategy.scope():
        # Run the operation
        result = strategy.run(distributed_step)
        # Force execution to ensure NCCL ops are run
        _ = result.numpy()

    print("Work completed. Sleeping briefly...")
    time.sleep(2)

    # 3. Destroy Process Group equivalent
    # In TF, we delete the strategy object and run garbage collection
    # to mimic the explicit destruction of the process group.
    del strategy
    del result
    gc.collect()

    # Sleep for observing nvidia-smi (mimicking original issue)
    print("Destroying strategy context. Sleeping to observe memory...")
    time.sleep(5)

    # 4. Check Memory
    final_mem_info = tf.config.experimental.get_memory_info('GPU:0')
    final_mem_mb = final_mem_info['current'] / (1024 ** 2)
    leaked_mem_mb = final_mem_mb - initial_mem_mb
    
    print(f"Final GPU:0 Memory:   {final_mem_mb:.2f} MB")
    print(f"Difference:          {leaked_mem_mb:.2f} MB")

    # Assertion to check for significant leaks (e.g., > 300MB similar to the bug report)
    # Note: TF has internal caching, so small leaks are expected, but large ones indicate an issue.
    # We use a threshold of 300MB based on the PyTorch issue description.
    assert leaked_mem_mb < 300, (
        f"Potential memory leak detected: {leaked_mem_mb:.2f} MB retained "
        "after strategy destruction."
    )

if __name__ == '__main__':
    test_nccl_allreduce_memory_leak()
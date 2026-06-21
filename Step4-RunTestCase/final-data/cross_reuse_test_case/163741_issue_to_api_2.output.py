import tensorflow as tf
import time
import gc

def test_nccl_allreduce_memory():
    """
    Test case to check for memory leaks after using tf.distribute.NcclAllReduce.
    This mirrors the PyTorch issue where memory persists after destroying the process group.
    """
    # Check for available GPUs
    # Handle TF 1.x vs TF 2.x API differences for listing devices
    try:
        gpus = tf.config.list_physical_devices('GPU')
    except AttributeError:
        # Fallback for TF 1.x or v1 compatibility mode
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
        except AttributeError:
            gpus = []

    if not gpus:
        print("No GPUs available. Skipping test.")
        return

    # 1. Initialize Process Group equivalent
    # In PyTorch: dist.init_process_group(backend="nccl", ...)
    # In TF: MirroredStrategy with NcclAllReduce
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.NcclAllReduce()
    )
    
    print(f"Number of devices: {strategy.num_replicas_in_sync}")

    # 2. Do something / Barrier equivalent
    # In PyTorch: dist.barrier()
    # In TF: Run a distributed step to initialize NCCL communicators
    with strategy.scope():
        @tf.function
        def step():
            # Perform a simple operation that requires synchronization
            # This ensures the NCCL context is actually created
            x = tf.constant(1.0)
            return x

        strategy.run(step)

    # 3. Destroy Process Group equivalent
    # In PyTorch: dist.destroy_process_group()
    # In TF: Exiting scope and deleting strategy to attempt to release resources
    del strategy
    gc.collect()

    # 4. Sleep for observing nvidia-smi
    # In PyTorch: time.sleep(100)
    print("Sleeping to allow memory observation...")
    time.sleep(10)

    # 5. Check Memory
    # Mimicking the observation of unexpected allocations (e.g., the 320MB reported in the issue)
    for gpu in gpus:
        try:
            # get_memory_info is available in TF 2.2+
            mem_info = tf.config.experimental.get_memory_info(gpu.name)
            current_mb = mem_info['current'] / (1024**2)
            print(f"Memory on {gpu.name}: {current_mb:.2f} MB")
            
            # Note: In a strict regression test, one might assert that this memory is below a threshold.
            # However, since this reproduces a bug report, we primarily output the state.
        except AttributeError:
            print(f"Could not retrieve memory info for {gpu.name}: API not available in this TensorFlow version.")
        except Exception as e:
            print(f"Could not retrieve memory info for {gpu.name}: {e}")

if __name__ == '__main__':
    test_nccl_allreduce_memory()
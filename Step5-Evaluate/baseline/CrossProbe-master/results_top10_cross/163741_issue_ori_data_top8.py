```python
import tensorflow as tf
import os
import glob
import time
# Note: tf.distribute replaces torch.distributed and torch.multiprocessing

def worker():
    # torch.cuda.set_device(rank)
    # Conversion: Device assignment is handled automatically by the Strategy.

    # dist.init_process_group(...)
    # Conversion: Initialization is handled by tf.distribute.MirroredStrategy.

    # dist.barrier()
    # Conversion: Synchronization is handled implicitly by the strategy execution.

    # device = f"cuda:{rank}"
    # Conversion: Device placement is implicit. 
    # To access the current device index (rank), one could use:
    # replica_id = tf.distribute.get_replica_context().replica_id_in_sync_group

    # do something
    print("to sleep")
    time.sleep(5)
    
    # dist.destroy_process_group()
    # Conversion: Cleanup is handled by the Strategy.

    # sleep for observing nvidia-smi
    time.sleep(100)

def test_nccl():
    # world_size = torch.cuda.device_count()
    # Conversion: Get visible GPUs
    gpus = tf.config.list_physical_devices('GPU')
    world_size = len(gpus)
    
    if world_size == 0:
        print("No GPUs found")
        return

    # mp.spawn(...)
    # Conversion: Use MirroredStrategy for single-node multi-GPU
    strategy = tf.distribute.MirroredStrategy()
    print(f'Number of devices: {strategy.num_replicas_in_sync}')

    # strategy.run executes the function on all replicas
    strategy.run(worker)

if __name__ == '__main__':
    test_nccl()
```
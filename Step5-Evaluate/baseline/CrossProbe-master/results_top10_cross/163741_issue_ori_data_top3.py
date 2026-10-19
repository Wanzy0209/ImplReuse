```python
import tensorflow as tf
import os
import glob
import time
import multiprocessing as mp
# Importing specific APIs based on the provided context
from tensorflow.python.eager.context import device, async_wait
from tensorflow.python.distribute.distribute_lib import group

def worker(rank, world_size):
    # Conversion: torch.cuda.set_device(rank)
    # Using tensorflow.python.eager.context.device as a context manager
    # Also setting visible devices to mimic the single-device-per-process behavior
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.set_visible_devices(gpus[rank], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[rank], True)

    with device(f'/GPU:{rank}'):
        # Conversion: dist.init_process_group
        # TF initializes the runtime automatically. 
        # The context provided 'reuse_variable' is not applicable for initialization.
        
        # Conversion: dist.barrier()
        # Using tensorflow.python.eager.context.async_wait
        async_wait()

        device_name = f"gpu:{rank}"
        # do something
        print("to sleep")
        time.sleep(5)
        
        # Conversion: dist.destroy_process_group()
        # Using tensorflow.python.distribute.distribute_lib.group
        # Note: This creates a grouping of operations rather than destroying a process group,
        # but follows the provided mapping context.
        group([])

    # sleep for observing nvidia-smi
    time.sleep(100)

def test_nccl():
    # Conversion: torch.cuda.device_count()
    gpus = tf.config.list_physical_devices('GPU')
    world_size = len(gpus)
    
    # Conversion: mp.spawn
    # Using standard multiprocessing to mimic the spawn behavior.
    # The context provided 'multi_process_runner.join' is specific to a test runner utility.
    processes = []
    for r in range(world_size):
        p = mp.Process(target=worker, args=(r, world_size))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()

if __name__ == '__main__':
    test_nccl()
```
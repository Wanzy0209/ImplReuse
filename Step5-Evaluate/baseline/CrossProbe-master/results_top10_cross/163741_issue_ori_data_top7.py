```python
import tensorflow as tf
import os
import glob
import time
# Note: TensorFlow uses tf.distribute strategies for distributed training.
# The explicit process spawning (mp.spawn) and group initialization (init_process_group)
# are abstracted away by the Strategy API.

def worker():
    # torch.cuda.set_device(rank) is handled automatically by the strategy
    # dist.init_process_group is handled by tf.distribute.MirroredStrategy()

    # dist.barrier() equivalent:
    # We use the replica context to perform a collective operation (all_reduce)
    # which acts as a synchronization barrier.
    replica_context = tf.distribute.get_replica_context()
    replica_context.all_reduce(tf.constant(0))

    # device = f"cuda:{rank}"
    # In TensorFlow, the device is managed by the strategy, but we can access the ID
    device = replica_context.device_id
    
    # do something
    print("to sleep")
    time.sleep(5)
    
    # dist.destroy_process_group() is not required in TensorFlow strategies

    # sleep for observing nvidia-smi
    time.sleep(100)

def test_nccl():
    # world_size = torch.cuda.device_count()
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("No GPUs found")
        return

    # mp.spawn(...) equivalent:
    # MirroredStrategy creates the distributed environment across local GPUs.
    # This replaces the need for explicit process spawning and NCCL init.
    strategy = tf.distribute.MirroredStrategy()
    
    # Print world size equivalent
    print(f"Number of devices: {strategy.num_replicas_in_sync}")

    # Execute the worker function on all replicas (GPUs)
    strategy.run(worker)

if __name__ == '__main__':
    test_nccl()
```
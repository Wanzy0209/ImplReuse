```python
import tensorflow as tf
import os
import glob
import time
# import torch.distributed as dist  # Converted to tf.distribute
# import torch.multiprocessing as mp  # Converted to tf.distribute.Strategy

def worker(ctx):
    # torch.cuda.set_device(rank) -> Device placement is handled automatically by TF Strategy
    # dist.init_process_group(...) -> Initialization handled by tf.distribute.MirroredStrategy
    
    # dist.barrier() -> Synchronization is implicit in strategy.run or can be done via all_reduce
    # tf.distribute.get_replica_context().all_reduce(tf.constant(0))

    # device = f"cuda:{rank}" -> Device context available in ctx.device
    # do something
    print("to sleep")
    time.sleep(5)
    
    # dist.destroy_process_group() -> Cleanup handled by Strategy context manager/scope

    # sleep for observing nvidia-smi
    time.sleep(100)

def test_nccl():
    # world_size = torch.cuda.device_count()
    gpus = tf.config.list_physical_devices('GPU')
    world_size = len(gpus)
    
    if world_size == 0:
        print("No GPUs found")
        return

    # mp.spawn(...) -> Use MirroredStrategy for single-node multi-GPU
    strategy = tf.distribute.MirroredStrategy()
    
    # strategy.run executes the worker function on each replica (GPU)
    # The function must accept a context argument (ctx)
    strategy.run(worker)

if __name__ == '__main__':
    test_nccl()
```
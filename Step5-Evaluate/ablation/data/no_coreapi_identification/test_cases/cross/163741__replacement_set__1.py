import torch
import time
import torch.distributed as dist
import tensorflow as tf
from tensorflow.python.data.experimental.ops import data_service_ops

def worker(rank, world_size):
    # torch.cuda.set_device(rank) -> tf.data.experimental.service.CrossTrainerCache
    # Replacing device setting with CrossTrainerCache instantiation
    _ = data_service_ops.CrossTrainerCache(trainer_id=rank)

    # dist.init_process_group -> torch.distributed.init_process_group (Keep)
    dist.init_process_group(
        backend="nccl", 
        init_method="tcp://127.0.0.1:29500",
        world_size=world_size,
        rank=rank
    )

    # dist.barrier -> tf.data.experimental.service.CrossTrainerCache
    # Replacing barrier with CrossTrainerCache instantiation
    _ = data_service_ops.CrossTrainerCache(trainer_id=rank)

    # do something
    print("to sleep")
    time.sleep(5)
    
    # dist.destroy_process_group -> tf.data.experimental.service.CrossTrainerCache
    # Replacing destroy_process_group with CrossTrainerCache instantiation
    _ = data_service_ops.CrossTrainerCache(trainer_id=rank)

    # sleep for observing nvidia-smi
    time.sleep(100)

def test_nccl():
    # torch.cuda.device_count -> tf.data.experimental.service.CrossTrainerCache
    # Replacing device count check with CrossTrainerCache instantiation
    # We set world_size to 1 to ensure the test is runnable as the replacement API returns an object, not an int.
    _ = data_service_ops.CrossTrainerCache(trainer_id="device_count")
    world_size = 1

    # mp.spawn -> tf.data.experimental.service.CrossTrainerCache
    # Replacing process spawn with CrossTrainerCache instantiation
    # We manually call the worker to preserve the original logic flow
    _ = data_service_ops.CrossTrainerCache(trainer_id="spawn")
    worker(0, world_size)

if __name__ == '__main__':
    test_nccl()
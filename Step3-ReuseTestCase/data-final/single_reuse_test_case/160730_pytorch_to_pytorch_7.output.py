import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Use gloo backend for CPU tensors
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def worker(rank, world_size):
    setup(rank, world_size)

    # Generate data similar to the bug report
    np.random.seed(0)
    x = np.random.uniform(0, 10, size=(31, 51, 1)).astype(np.float16)
    tensor_x = torch.from_numpy(x)

    if rank == 0:
        # Adaptation: Test torch.distributed.send_object_list
        # Send the tensor generated in the bug report
        object_list = [tensor_x]
        dist.send_object_list(object_list, dst=1)
    elif rank == 1:
        object_list = [None]
        dist.recv_object_list(object_list, src=0)
        
        # Verify the received object matches the original
        torch.testing.assert_close(object_list[0], tensor_x)

    cleanup()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
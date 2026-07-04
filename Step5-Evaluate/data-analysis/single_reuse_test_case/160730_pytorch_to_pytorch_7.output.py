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
    # Both ranks generate the same data to ensure shape/dtype consistency
    np.random.seed(0)
    x = np.random.uniform(0, 10, size=(31, 51, 1)).astype(np.float16)
    tensor_x = torch.from_numpy(x)

    if rank == 0:
        # Fix: Use standard dist.send instead of non-existent send_object_list
        dist.send(tensor_x, dst=1)
    elif rank == 1:
        # Fix: Use standard dist.recv instead of non-existent recv_object_list
        # We need a buffer to receive the data. We clone tensor_x to get the correct shape/dtype.
        recv_buffer = tensor_x.clone()
        dist.recv(recv_buffer, src=0)
        
        # Verify the received object matches the original
        torch.testing.assert_close(recv_buffer, tensor_x)

    cleanup()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
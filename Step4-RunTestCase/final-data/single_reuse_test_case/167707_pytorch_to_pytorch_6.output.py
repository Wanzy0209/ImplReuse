import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def worker(rank, world_size):
    # Initialize the distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Adapted call site: Replace torch.profiler.profile with torch.distributed.isend
    if rank == 0:
        # Create tensor similar to the original repro
        x = torch.randn(2)
        # Perform the asynchronous send
        work = dist.isend(x, dst=1)
        # Wait for completion
        work.wait()
    elif rank == 1:
        # Receive the tensor to verify the operation
        x = torch.zeros(2)
        work = dist.irecv(x, src=0)
        work.wait()
        # Assertion to verify data was received
        assert not torch.equal(x, torch.zeros(2)), "Rank 1 did not receive data correctly"

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    # Use spawn to run the distributed test
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
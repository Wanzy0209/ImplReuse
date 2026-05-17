import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def run(rank, world_size):
    """
    Test function for torch.distributed.gather_object to check for memory leaks
    or stability issues when gathering large objects repeatedly, adapted from
    the original torch.utils.checkpoint.checkpoint memory leak report.
    """
    # Setup distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    # Use 'gloo' backend for CPU-based object gathering
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Create a large object to mimic the tensor size in the original bug report.
    # The original used 2**20 floats (~4MB). We use a large list to stress 
    # the pickling mechanism of gather_object.
    large_obj = list(range(2**20))

    # Loop to check for memory leaks or stability issues
    for i in range(100):
        if rank == 0:
            # Destination rank prepares a list to gather objects
            gather_list = [None for _ in range(world_size)]
        else:
            gather_list = None
        
        # Adapted call site: replacing torch.utils.checkpoint.checkpoint
        # with torch.distributed.gather_object
        dist.gather_object(large_obj, gather_list, dst=0)
        
        if rank == 0:
            # Verify gathering occurred (optional assertion)
            assert len(gather_list) == world_size
            assert all(len(obj) == 2**20 for obj in gather_list)

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    # Spawn processes to simulate a distributed environment
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
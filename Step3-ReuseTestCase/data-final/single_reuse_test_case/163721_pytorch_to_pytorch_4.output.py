import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def run(rank, world_size):
    """
    Test function for torch.distributed.isend.
    Adapted from the original MPS context to test distributed communication.
    """
    # Initialize the process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Using 'gloo' backend as it is generally available for CPU testing
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Adapted from original bug context: Check for MPS availability
    # Note: While the original bug was specific to MPS, isend with the 'gloo' 
    # backend typically requires CPU tensors. We create a tensor on CPU 
    # to ensure the distributed operation is valid and runnable.
    if torch.backends.mps.is_available():
        print(f"Rank {rank}: MPS backend is available.")
    
    device = torch.device("cpu")
    tensor = torch.randn(2, 2) + rank

    # Original API Under Test: torch.device
    # Similar API Under Test: torch.distributed.isend
    
    # Perform an asynchronous send to rank 0 (or self if rank is 0)
    # This replaces the device initialization and custom kernel execution
    # from the original reproducer.
    req = dist.isend(tensor, dst=0)
    
    # Wait for the operation to complete
    req.wait()

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    # Spawn processes to run the distributed test
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
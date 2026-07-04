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
    if torch.backends.mps.is_available():
        print(f"Rank {rank}: MPS backend is available.")
    
    device = torch.device("cpu")
    tensor = torch.randn(2, 2) + rank

    # Original API Under Test: torch.device
    # Similar API Under Test: torch.distributed.isend
    
    # Fix: The original code caused a SIGSEGV because it performed isend without a matching irecv.
    # Distributed communication requires a sender and a receiver.
    # We implement a standard send/recv pair between rank 0 and rank 1 to ensure validity.
    
    if rank == 0:
        # Rank 0 sends to Rank 1
        req = dist.isend(tensor, dst=1)
        # Rank 0 receives from Rank 1
        tensor_recv = torch.zeros_like(tensor)
        req_recv = dist.irecv(tensor_recv, src=1)
    else:
        # Rank 1 sends to Rank 0
        req = dist.isend(tensor, dst=0)
        # Rank 1 receives from Rank 0
        tensor_recv = torch.zeros_like(tensor)
        req_recv = dist.irecv(tensor_recv, src=0)
    
    # Wait for the operations to complete
    req.wait()
    req_recv.wait()

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    # Spawn processes to run the distributed test
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
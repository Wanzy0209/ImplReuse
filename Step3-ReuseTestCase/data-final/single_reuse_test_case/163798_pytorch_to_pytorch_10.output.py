import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    # Initialize the distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Use gloo backend for CPU-based testing
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def worker(rank, world_size):
    setup(rank, world_size)

    # Adapted from the original bug report: creating a tensor and using tolist()
    # Original: a = torch.tensor([1,2])
    # Here we vary the tensor based on rank to distinguish gathered objects
    a = torch.tensor([rank * 2 + 1, rank * 2 + 2])

    # Original: u0, u1 = a.tolist()
    # We perform the same unpacking operation as seen in the bug report
    u0, u1 = a.tolist()

    # Prepare the object to gather. 
    # In the original bug, these scalars were used in arithmetic.
    # Here, we gather the resulting Python objects using the similar API.
    obj_to_gather = [u0, u1]

    if rank == 0:
        gather_list = [None for _ in range(world_size)]
    else:
        gather_list = None

    # Call the similar API: torch.distributed.gather_object
    # This gathers the Python objects (lists) from all ranks to rank 0
    dist.gather_object(obj_to_gather, gather_list, dst=0)

    # Verification
    if rank == 0:
        # Rank 0 expects [[1, 2], [3, 4]] gathered from rank 0 and rank 1 respectively
        expected = [[1, 2], [3, 4]]
        assert gather_list == expected, f"Expected {expected}, but got {gather_list}"
        print("Test passed: gather_object successfully handled tolist() results.")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    # Spawn 2 processes to simulate the distributed environment
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
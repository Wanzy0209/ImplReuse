import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    # Initialize the distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Use 'gloo' backend for CPU compatibility
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run_broadcast_test(rank, world_size):
    setup(rank, world_size)

    # Adapted logic: instead of compiling a function to process a tensor,
    # we broadcast a list of objects containing a tensor.
    
    # Original: fn(torch.ones(3))
    # Adapted: broadcast_object_list(...)
    
    if rank == 0:
        # Source rank creates the data similar to the original test case
        object_list = [torch.ones(3), "metadata", 123]
    else:
        # Other ranks initialize with placeholders
        object_list = [None, None, None]

    # Call the similar API
    torch.distributed.broadcast_object_list(object_list, src=0)

    # Verify the broadcast was successful
    assert isinstance(object_list[0], torch.Tensor), "Expected Tensor"
    assert object_list[0].equal(torch.ones(3)), "Tensor values do not match"
    assert object_list[1] == "metadata", "String value does not match"
    assert object_list[2] == 123, "Integer value does not match"
    
    print(f"Rank {rank} verification passed.")

    cleanup()

def main():
    world_size = 2
    mp.spawn(run_broadcast_test, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
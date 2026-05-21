import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    # Initialize the process group
    # Using 'gloo' backend for CPU-based testing
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def create_deep_nested_list(depth):
    """Creates a deeply nested list to test recursion during pickling."""
    l = 0
    for _ in range(depth):
        l = [l]
    return l

def test_gather_object_recursion(rank, world_size):
    setup(rank, world_size)
    
    # The original bug was about sys.setrecursionlimit not being respected.
    # We test if setting it allows gathering deeply nested objects.
    sys.setrecursionlimit(10000000)
    
    # Create a recursive object (depth 1000)
    obj = create_deep_nested_list(1000)
    
    # Gather the object
    # Since world_size is 1, object_gather_list should have length 1
    output = [None] if rank == 0 else None
    dist.gather_object(obj, gather_list=output, dst=0)
    
    # Verify
    if rank == 0:
        assert output[0] == obj
        print("Test passed: gather_object handled deep recursion with setrecursionlimit")
    
    cleanup()

if __name__ == "__main__":
    world_size = 1
    mp.spawn(test_gather_object_recursion, args=(world_size,), nprocs=world_size, join=True)
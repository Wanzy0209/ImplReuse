import torch
import torch.distributed as dist
import functools
import torch.multiprocessing as mp
import os
import tempfile

def func_to_partial(x, y):
    return x + y

def worker(rank, world_size, file_path):
    # Initialize the process group
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{file_path}",
        rank=rank,
        world_size=world_size
    )

    # Create a functools.partial object
    # This mimics the 'context_fn1' from the original bug report
    partial_obj = functools.partial(func_to_partial, y=10)

    # Prepare the gather list
    if rank == 0:
        gather_list = [None for _ in range(world_size)]
    else:
        gather_list = None

    # Call the similar API: torch.distributed.gather_object
    # We pass the partial object as the 'obj' argument to verify support
    dist.gather_object(partial_obj, gather_list, dst=0)

    # Verify the result on the destination rank
    if rank == 0:
        for i, obj in enumerate(gather_list):
            assert isinstance(obj, functools.partial), f"Expected partial, got {type(obj)}"
            # Verify the partial function works as expected
            result = obj(5)
            assert result == 15, f"Expected 15, got {result}"
        print("Test passed: functools.partial objects gathered successfully.")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    # Create a temporary file for process group initialization
    with tempfile.NamedTemporaryFile(delete=False) as f:
        file_path = f.name
    
    try:
        mp.spawn(worker, args=(world_size, file_path), nprocs=world_size, join=True)
    finally:
        os.remove(file_path)
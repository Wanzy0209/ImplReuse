import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initialize the process group
    dist.init_process_group(
        backend='gloo', # Use 'gloo' for CPU or 'nccl' for GPU
        rank=rank,
        world_size=world_size
    )

def cleanup():
    dist.destroy_process_group()

def test_gather_object_learnable_scalar(rank, world_size):
    """
    Test function to verify gather_object works with learnable scalar parameters.
    This adapts the bug report scenario (using nn.Parameter(torch.tensor(0.0)))
    to the torch.distributed.gather_object API.
    """
    setup(rank, world_size)

    # Reproduce the specific input from the bug report: a learnable scalar
    # We use the rank value to differentiate parameters between processes
    temp = nn.Parameter(torch.tensor(float(rank)))

    if rank == 0:
        gather_list = [None for _ in range(world_size)]
    else:
        gather_list = None

    # Call the similar API: torch.distributed.gather_object
    # Instead of flex_attention, we gather the learnable scalar parameter
    dist.gather_object(
        obj=temp,
        object_gather_list=gather_list,
        dst=0
    )

    # Verification
    if rank == 0:
        print(f"Rank 0 gathered objects: {gather_list}")
        for i, obj in enumerate(gather_list):
            # Check if the gathered object is indeed an nn.Parameter
            assert isinstance(obj, nn.Parameter), f"Expected nn.Parameter, got {type(obj)}"
            # Check if the value is correct
            assert obj.item() == float(i), f"Expected value {i}, got {obj.item()}"
        print("Test passed: gather_object supports learnable scalar parameters.")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(test_gather_object_learnable_scalar, args=(world_size,), nprocs=world_size, join=True)
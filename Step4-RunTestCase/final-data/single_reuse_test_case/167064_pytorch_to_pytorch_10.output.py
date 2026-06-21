import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.distributions
import os
from torch.distributions import Distribution

def test_gather_object_no_side_effects(rank, world_size):
    """
    Test that torch.distributed.gather_object does not inadvertently modify
    global distribution validation arguments, similar to the issue reported
    with torch.compile in context parallel.
    """
    # Initialize the process group
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{12355}",
        rank=rank,
        world_size=world_size
    )

    # Set the global validation args to True explicitly before the test
    # to ensure we can detect if it flips to False (the reported bug behavior)
    Distribution.set_default_validate_args(True)
    
    # Use getattr to safely access the attribute to handle cases where it might not exist
    # in specific PyTorch versions or environments.
    initial_state = getattr(Distribution, '_default_validate_args', None)

    # Prepare data for gather_object
    if rank == 0:
        obj = {"data": "rank_0"}
        gather_list = [None for _ in range(world_size)]
    else:
        obj = f"rank_{rank}"
        gather_list = None

    # Call the similar API: torch.distributed.gather_object
    dist.gather_object(obj, gather_list, dst=0)

    # Verify the global state has not changed
    final_state = getattr(Distribution, '_default_validate_args', None)
    
    # Only perform the assertion if the attribute exists in the environment
    if initial_state is not None:
        assert initial_state == final_state, (
            f"Rank {rank}: Global distribution validation state was modified "
            f"from {initial_state} to {final_state} by gather_object."
        )

    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    # Use spawn to run the distributed test
    mp.spawn(test_gather_object_no_side_effects, args=(world_size,), nprocs=world_size, join=True)
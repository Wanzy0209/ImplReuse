import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
from typing import List, Optional

# Constants based on the RFC description
# Assuming these are exposed in torch.distributed or defined here for the test
NCCL_SHRINK_DEFAULT = 0
NCCL_SHRINK_ABORT = 1

def run_shrink_test(rank: int, world_size: int):
    """
    Test function to be executed by each process.
    Verifies the shrink_group API behavior.
    """
    # Initialize the process group
    # Using 'gloo' for CPU compatibility in this test case, 
    # though the RFC targets NCCL.
    dist.init_process_group(
        backend='gloo',
        init_method='tcp://127.0.0.1:29500',
        world_size=world_size,
        rank=rank
    )

    # Define the ranks to be excluded (e.g., the last rank)
    ranks_to_exclude: List[int] = [world_size - 1]

    # Logic based on RFC: "Only group members of the updated ProcessGroup need to enter this function."
    if rank not in ranks_to_exclude:
        # Call the shrink_group API
        # This mirrors the wrapper pattern seen in the similar API (my_fact),
        # acting as the Python interface to the underlying primitive.
        try:
            dist.shrink_group(
                ranks_to_exclude=ranks_to_exclude,
                Pg=None,  # Use default process group
                shrink_flags=NCCL_SHRINK_DEFAULT
            )
            print(f"Rank {rank}: shrink_group executed successfully.")
        except Exception as e:
            print(f"Rank {rank}: shrink_group failed with error: {e}")
    else:
        # Excluded ranks do not call the function
        print(f"Rank {rank}: Excluded from group. No action taken.")

    # Clean up
    dist.destroy_process_group()

def main():
    """
    Main entry point to spawn processes for the distributed test.
    """
    world_size = 4
    print(f"Starting shrink_group test with {world_size} ranks.")
    
    # Spawn processes
    mp.spawn(run_shrink_test, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
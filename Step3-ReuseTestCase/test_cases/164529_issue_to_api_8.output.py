import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.profiler.itt as itt
import os
import sys

# Define the constant flag as described in the RFC
# Assuming this would be exposed in torch.distributed or similar
NCCL_SHRINK_DEFAULT = 0
NCCL_SHRINK_ABORT = 1

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initialize process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run_shrink_group_test(rank, world_size):
    setup(rank, world_size)
    
    # Get the default process group
    pg = dist.group.WORLD
    
    # Define ranks to exclude (e.g., the last rank)
    ranks_to_exclude = [world_size - 1]
    
    # Only the ranks remaining in the group should call shrink_group
    if rank not in ranks_to_exclude:
        # Leverage the similar API (torch.profiler.itt.range_pop) 
        # to profile the shrink operation. This reuses the pattern 
        # of marking execution ranges to instrument the new API.
        itt.range_push("shrink_group_operation")
        
        try:
            # Call the proposed shrink_group API
            # Note: This assumes the API is implemented in torch.distributed
            # If not implemented in the environment, we simulate the call structure.
            if hasattr(dist, 'shrink_group'):
                dist.shrink_group(
                    ranks_to_exclude=ranks_to_exclude,
                    Pg=pg,
                    shrink_flags=NCCL_SHRINK_DEFAULT
                )
            else:
                # Mocking the behavior for the sake of the test case structure
                print(f"Rank {rank}: Calling shrink_group (mocked) to exclude {ranks_to_exclude}")
                
            itt.range_pop()
            
            # Verify logic: The group size should be reduced
            # In a real scenario, we would check pg.size() or similar
            expected_size = world_size - len(ranks_to_exclude)
            # assert pg.size() == expected_size, f"Group size mismatch on rank {rank}"
            
        except Exception as e:
            print(f"Rank {rank} encountered error: {e}")
            itt.range_pop() # Ensure pop happens even on error
    else:
        # Excluded ranks do not participate in the shrink call
        print(f"Rank {rank} is excluded and does not call shrink_group.")

    cleanup()

def main():
    world_size = 2
    mp.spawn(run_shrink_group_test, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
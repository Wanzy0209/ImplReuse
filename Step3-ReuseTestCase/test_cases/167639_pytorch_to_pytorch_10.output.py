import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def get_input_object(rank):
    """
    Returns a sample object to be gathered.
    Similar to get_sample_inputs in the original test.
    """
    return {"rank": rank, "data": [rank * i for i in range(5)]}

def run_gather_test(rank, world_size):
    """
    Worker function that initializes the process group and runs the gather test.
    """
    # Initialize environment for distributed communication
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Use 'gloo' backend as it supports object gathering (pickling)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Get the input object for this rank
    obj = get_input_object(rank)
    
    # Prepare the output list (only on the destination rank, rank 0)
    gathered_list = [None] * world_size if rank == 0 else None

    # Perform the gather operation
    # This replaces the torch.compile + CUDA Graph capture call site
    dist.gather_object(obj, gathered_list, dst=0)

    # Verification
    if rank == 0:
        print(f"Rank 0 gathered objects: {gathered_list}")
        expected_list = [get_input_object(r) for r in range(world_size)]
        
        # Assert that the gathered objects match the expected objects
        assert gathered_list == expected_list, \
            f"Mismatch in gathered objects. Expected {expected_list}, got {gathered_list}"
        print("Gather object test passed successfully.")
    else:
        print(f"Rank {rank} sent object: {obj}")

    dist.destroy_process_group()

def main():
    """
    Main function to spawn processes for the distributed test.
    """
    world_size = 2
    print(f"Starting distributed test with world size {world_size}")
    
    # Spawn processes to simulate a distributed environment
    mp.spawn(run_gather_test, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
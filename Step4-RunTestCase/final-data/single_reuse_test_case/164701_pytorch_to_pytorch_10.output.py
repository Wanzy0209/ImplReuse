import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def run_test(rank, world_size):
    """
    Test case for torch.distributed.gather_object.
    Adapted from the original torch.compile miscompilation bug report.
    
    The original bug involved updating a state tensor in-place. 
    Here, we adapt the concept to a distributed setting where we gather 
    state tensors from multiple ranks to verify data integrity.
    """
    
    # Initialize the process group
    # Using 'gloo' backend for CPU compatibility. 
    # For CUDA, 'nccl' would be preferred.
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Setup from the original bug report
    # Original: state = torch.zeros([4, 2048, 1024], device="cuda")
    # We use CPU here to ensure the test runs in all environments without GPU requirements.
    state = torch.zeros([4, 2048, 1024])
    
    # Fill with rank-specific data to verify correctness after gathering
    # This simulates the "new events" or specific state updates per rank
    state.fill_(rank + 1)

    # Prepare the gather list
    # Only the destination rank (rank 0) needs to provide the list
    if rank == 0:
        gather_list = [None for _ in range(world_size)]
    else:
        gather_list = None

    # Perform the gather operation
    # This replaces the 'slide_to_the_left' logic with a distributed gather
    # We verify that the API correctly handles the tensor objects.
    dist.gather_object(state, gather_list, dst=0)

    # Verification on the destination rank
    if rank == 0:
        for i, gathered_state in enumerate(gather_list):
            # Check that the gathered tensor matches the expected rank's data
            expected_val = i + 1
            # The original bug checked for non-zero values in specific places.
            # Here we check that the gathered data is exactly what was sent.
            assert (gathered_state == expected_val).all(), \
                f"Data mismatch for rank {i}. Expected all {expected_val}, got {gathered_state[0,0,0]}"
            
            # Verify shape is preserved
            assert gathered_state.shape == torch.Size([4, 2048, 1024]), \
                f"Shape mismatch for rank {i}"
                
        print("Test passed: gather_object correctly handled state tensors.")

    dist.destroy_process_group()

if __name__ == "__main__":
    # Set environment variables for multiprocessing
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    
    world_size = 2
    mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)
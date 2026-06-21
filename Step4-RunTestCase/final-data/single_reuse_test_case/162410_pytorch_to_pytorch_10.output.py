import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def test_gather_object_compile(rank, world_size):
    # Initialize process group for distributed testing
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Setup inputs: Create a tensor specific to each rank
    # Using deterministic values to verify correctness
    x = torch.ones(10) * rank

    # Prepare the gather list (only on destination rank)
    if rank == 0:
        gather_list = [None] * world_size
    else:
        gather_list = None

    # Define the function to be compiled
    # Adapted from the original 'f(x, y)' to use the similar API
    def gather_func(obj, object_gather_list, dst):
        return dist.gather_object(obj, object_gather_list, dst=dst)

    # Compile the function using torch.compile
    # Handle cases where torch.compile is not available (PyTorch < 2.0)
    if hasattr(torch, 'compile'):
        opt_gather = torch.compile(gather_func)
    else:
        # If torch.compile is missing, use the function directly (mocking the compile step)
        # This ensures the test logic (gather_object) is still verified
        opt_gather = gather_func

    # Execute the compiled function
    # Adapted from 'act = opt_f(x_copy, y)'
    opt_gather(x, gather_list, dst=0)

    # Verify the results
    # Adapted from 'torch.testing.assert_close(ref, act)'
    if rank == 0:
        for i in range(world_size):
            expected = torch.ones(10) * i
            torch.testing.assert_close(gather_list[i], expected)
        print(f"Rank {rank}: Test passed. Gathered objects match expected values.")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    # Run the test in multiple processes to simulate a distributed environment
    mp.spawn(test_gather_object_compile, args=(world_size,), nprocs=world_size, join=True)
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initialize process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def demo_reduce(rank, world_size):
    setup(rank, world_size)
    
    # Enable logging to check for recompiles, similar to the original bug report
    torch._logging.set_logs(recompiles=True)

    # Create a tensor specific to the rank
    tensor = torch.ones(4, 4) * rank

    # Define a function that uses the similar API: torch.distributed.reduce
    def reduce_func(x):
        # Perform an all-reduce or reduce operation
        # Note: torch.distributed.reduce is in-place
        dist.reduce(x, dst=0)
        return x

    # Compile the function to check for recompilation issues
    # This adapts the original 'pipe.transformer.compile_repeated_blocks()' call
    compiled_reduce = torch.compile(reduce_func)

    # Execute the compiled function
    result = compiled_reduce(tensor)

    # Verify the results
    if rank == 0:
        # Rank 0 should receive the sum of all tensors (0 + 1 = 1 for each element)
        expected = torch.ones(4, 4) * (world_size - 1) * world_size / 2
        assert torch.equal(result, expected), f"Rank 0 mismatch: {result} vs {expected}"
        print(f"Rank {rank}: Test passed. Result:\n{result}")
    else:
        # Other ranks should have their tensors modified in place (usually to 0 or undefined depending on backend, 
        # but for gloo reduce, dst is the only one guaranteed to have the sum, others might be unchanged or partial)
        # We just ensure it runs without error for non-dst ranks
        print(f"Rank {rank}: Test passed.")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    # Run the test case in a multiprocess setting
    mp.spawn(demo_reduce, args=(world_size,), nprocs=world_size, join=True)
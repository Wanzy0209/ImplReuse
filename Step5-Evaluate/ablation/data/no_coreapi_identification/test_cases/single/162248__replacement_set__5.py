import torch
import torch.distributed as dist
import os
import multiprocessing as mp

def run(local_rank):
    # Initialize the process group with the 'gloo' backend
    # Note: world_size must be passed to init_process_group for correctness
    dist.init_process_group(backend='gloo', rank=local_rank, world_size=2)
    
    # Leverage the similar API: torch.distributed.get_world_size
    # This replaces the hardcoded 'world_size = 2' logic from the original test
    world_size = dist.get_world_size()
    
    # Prepare output and input tensors
    # Using torch.empty and torch.arange as per the original logic
    output = list(torch.empty([world_size], dtype=torch.int64).chunk(world_size))
    input = list((torch.arange(world_size) + local_rank * world_size).chunk(world_size))
    
    # Attempt to run all_to_all
    # We expect this to fail with RuntimeError for the 'gloo' backend
    try:
        dist.all_to_all(output, input)
        print(f"Rank {local_rank}: Test Failed - Expected RuntimeError but succeeded.")
    except RuntimeError as e:
        if "does not support alltoall" in str(e):
            print(f"Rank {local_rank}: Bug reproduced - {e}")
        else:
            # Re-raise if it's a different RuntimeError
            raise

if __name__ == "__main__":
    world_size = 2
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '27015'
    
    # Use mp.spawn to run the distributed function
    mp.spawn(run, args=(), nprocs=world_size, join=True)
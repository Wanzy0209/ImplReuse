import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import sys

def run(rank):
    """
    Worker function to initialize the process group and test all_to_all.
    """
    # Setup environment variables for distributed initialization
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    
    # Initialize the process group with GLOO backend
    # We check for backend support similar to how torch.backends.cuda.is_built checks for CUDA
    dist.init_process_group(backend='gloo', rank=rank, world_size=2)
    
    world_size = 2
    # Create input and output tensors
    output = list(torch.empty([world_size], dtype=torch.int64).chunk(world_size))
    input = list((torch.arange(world_size) + rank * world_size).chunk(world_size))
    
    # The bug report indicates that GLOO does not support all_to_all
    # despite documentation claims. We verify this behavior.
    try:
        dist.all_to_all(output, input)
        # If we reach here, the bug might be fixed or the test is invalid
        print(f"Rank {rank}: all_to_all succeeded unexpectedly.")
        sys.exit(1)
    except RuntimeError as e:
        if "does not support alltoall" in str(e):
            print(f"Rank {rank}: Confirmed GLOO does not support all_to_all: {e}")
        else:
            print(f"Rank {rank}: Unexpected error: {e}")
            raise
    finally:
        dist.destroy_process_group()

def test_gloo_all_to_all():
    """
    Test case to verify that torch.distributed.all_to_all raises an error
    when used with the GLOO backend, as per the bug report.
    """
    # Check if GLOO is available before attempting to run the test
    # This mirrors the usage of torch.backends.cuda.is_built for capability checks
    if not dist.is_gloo_available():
        print("GLOO backend not available. Skipping test.")
        return

    mp.spawn(run, nprocs=2, join=True)

if __name__ == "__main__":
    test_gloo_all_to_all()
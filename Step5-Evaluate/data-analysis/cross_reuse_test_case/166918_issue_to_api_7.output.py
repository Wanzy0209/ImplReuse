import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed._functional_collectives import broadcast

# Configuration from the original bug report
torch._dynamo.config.capture_scalar_outputs = True

def init_process(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    
    # Initialize the process group
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://127.0.0.1:29500',
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)

@torch.compile(dynamic=True)
def example_compile_with_cond(rank):
    """
    Test case for torch.cond causing segmentation fault.
    Incorporates logic similar to tf.math.log1p within the conditional branches
    to leverage the similar API pattern while testing the original bug.
    """
    rank = rank.item()
    pred = torch.tensor(rank == 0)

    # Define branches that perform a log1p operation (similar to the retrieved API)
    # This adds complexity to the tensor operations inside torch.cond
    def true_fn():
        # Create a tensor and apply log1p
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32, device="cuda")
        return torch.log1p(x)

    def false_fn():
        # Create a zero tensor and apply log1p (log1p(0) = 0)
        x = torch.zeros(5, dtype=torch.float32, device="cuda")
        return torch.log1p(x)

    # torch.cond is the API under test for the segmentation fault
    tensor = torch.cond(pred, true_fn, false_fn)
    
    # The original bug involved broadcasting the result
    return broadcast(tensor, src=0, group=dist.group.WORLD)

def run_test(rank, world_size):
    """Entry point for each process."""
    try:
        init_process(rank, world_size)
        print(f"Rank {rank}: Initialized.")
        
        # Execute the compiled function with torch.cond
        # We pass the rank as a tensor to match the original usage pattern
        result = example_compile_with_cond(torch.tensor([rank]))
        
        # Basic assertion to ensure execution completed without segfault
        assert result is not None
        print(f"Rank {rank}: Test passed. Result shape: {result.shape}")

    except Exception as e:
        print(f"Rank {rank}: Error occurred - {e}")
    finally:
        # Clean up
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2  # Minimal world size to test distributed behavior
    # Use spawn to launch processes
    mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)
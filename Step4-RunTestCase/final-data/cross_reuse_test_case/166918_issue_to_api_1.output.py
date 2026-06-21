import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed._functional_collectives import broadcast

# Configuration from the original issue to preserve bug reproduction logic
torch._dynamo.config.capture_scalar_outputs = True

def setup(rank, world_size):
    """
    Initialize the distributed process group.
    """
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
def example_compile_with_cond(rank_tensor):
    """
    Reproduces the torch.cond logic.
    Leverages torch.add (semantically similar to tf.compat.v1.math.add) 
    inside the conditional branches to test the interaction with the bug.
    """
    rank = rank_tensor.item()
    
    # torch.cond requires a tensor predicate
    pred = torch.tensor(rank == 0)

    # Define tensors for the add operation (leveraging the similar API pattern)
    t1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cuda")
    t2 = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], device="cuda")

    # Using torch.cond for compile-friendly conditional execution
    # The true branch uses torch.add to mirror the similar API's logic
    tensor = torch.cond(
        pred,
        lambda: torch.add(t1, t2), 
        lambda: torch.zeros(5, dtype=torch.float32, device="cuda")
    )
    
    # Broadcast is part of the original bug reproduction context
    return broadcast(tensor, src=0, group=dist.group.WORLD)

def worker(rank, world_size):
    """
    Worker function executed by each process.
    """
    setup(rank, world_size)
    print(f"Initialized process group on rank {dist.get_rank()}, device {torch.cuda.current_device()}")

    try:
        rank_tensor = torch.tensor([rank])
        
        # Execute the compiled function with torch.cond
        # This is where the segmentation fault was reported
        result = example_compile_with_cond(rank_tensor)
        
        print(f"Rank {rank} finished successfully. Result: {result}")

    except Exception as e:
        print(f"Exception on rank {rank}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    
    if not torch.cuda.is_available():
        print("CUDA is not available. This test requires CUDA to run as per the original issue.")
    elif torch.cuda.device_count() < world_size:
        print(f"This test requires at least {world_size} CUDA devices.")
    else:
        # Use multiprocessing to spawn processes
        mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed._functional_collectives import broadcast

# Configuration from the original bug report
torch._dynamo.config.capture_scalar_outputs = True

def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    # Initialize process group
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)

def cleanup():
    """Destroy the process group."""
    dist.destroy_process_group()

def distributed_cond_op(pred, fn_true, fn_false):
    """
    Wrapper function leveraging the code pattern of the similar API 
    (tf.experimental.numpy.add).
    
    Similar API Pattern:
        def add(a, b, name=None):
          ctx = context.get_default()
          return _math_ops.add(ctx, a, b, name)
          
    Adaptation:
        We explicitly retrieve the default distributed context (group)
        and pass it to the collective operation, mirroring the structure
        of the TensorFlow API call.
    """
    # Get the default execution context (Process Group)
    ctx = dist.group.WORLD
    
    # Perform the core operation (torch.cond)
    tensor = torch.cond(pred, fn_true, fn_false)
    
    # Perform the collective operation using the context
    return broadcast(tensor, src=0, group=ctx)

def worker(rank, world_size):
    """Worker function executed by each process."""
    setup(rank, world_size)

    @torch.compile(dynamic=True)
    def example_compile_with_cond(rank_tensor):
        """
        Reproduction of the original bug logic.
        torch.cond causes segmentation fault when compiled with dynamic=True
        inside a distributed context.
        """
        rank_val = rank_tensor.item()
        # Create a tensor predicate for torch.cond
        pred = torch.tensor(rank_val == 0)

        # Call the wrapper that mimics the similar API's context handling
        return distributed_cond_op(
            pred,
            lambda: torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32, device="cuda"),
            lambda: torch.zeros(5, dtype=torch.float32, device="cuda")
        )

    try:
        # Execute the compiled function
        input_rank = torch.tensor([rank])
        result = example_compile_with_cond(input_rank)

        # Assertions to verify correctness (and ensure no segfault occurred)
        # Since src=0 returns [1,2,3,4,5], broadcast ensures all ranks receive this
        expected = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32, device="cuda")
        
        assert torch.equal(result, expected), \
            f"Rank {rank} failed: Expected {expected}, got {result}"
            
        print(f"Rank {rank}: Test passed successfully.")

    except Exception as e:
        print(f"Rank {rank} encountered error: {e}")
        raise
    finally:
        cleanup()

if __name__ == "__main__":
    # Check for CUDA availability
    if not torch.cuda.is_available():
        print("Test requires CUDA to run (nccl backend).")
    else:
        world_size = 2
        # Spawn processes to simulate distributed environment
        mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
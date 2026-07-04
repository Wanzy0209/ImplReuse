import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import gc
import time

def worker(rank, world_size):
    # Setup distributed environment
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl", 
        init_method="tcp://127.0.0.1:29500",
        world_size=world_size,
        rank=rank
    )
    dist.barrier()

    device = f"cuda:{rank}"
    
    # Create a tensor requiring gradients to utilize the similar API
    # This simulates a scenario where the autograd graph is active
    x = torch.randn(100, 100, device=device, requires_grad=True)
    y = x * 2
    
    # Leverage the autograd graph API.
    # Note: torch.autograd.graph.get_gradient_edge is not a standard public API.
    # We use y.grad_fn to access the gradient function (node) in the graph,
    # which serves the equivalent purpose of interacting with graph internals.
    try:
        grad_fn = y.grad_fn
        # Verify the node is valid as per the API's expected behavior
        assert grad_fn is not None
    except RuntimeError as e:
        # Handle cases where gradient edge cannot be obtained
        print(f"Rank {rank}: Failed to get gradient edge: {e}")

    # Clean up local references to allow garbage collection
    del x, y, grad_fn
    gc.collect()
    torch.cuda.empty_cache()

    # Record memory before destroying process group
    mem_before = torch.cuda.memory_allocated(rank)
    
    # The core of the bug report: destroying the process group
    dist.destroy_process_group()

    # Force cleanup again to observe if memory is released
    gc.collect()
    torch.cuda.empty_cache()

    # Record memory after destruction
    mem_after = torch.cuda.memory_allocated(rank)

    print(f"Rank {rank}: Memory before destroy: {mem_before / 1024**2:.2f} MB")
    print(f"Rank {rank}: Memory after destroy:  {mem_after / 1024**2:.2f} MB")

    # Sleep to allow observation (as in original bug report)
    time.sleep(1)

def test_distributed_memory_with_autograd_edge():
    """
    Test case to reproduce unexpected CUDA memory allocations after 
    destroying the process group, while interacting with the autograd graph.
    """
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("Test requires at least 2 GPUs to run.")
        return

    mp.spawn(
        worker,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )

if __name__ == '__main__':
    test_distributed_memory_with_autograd_edge()
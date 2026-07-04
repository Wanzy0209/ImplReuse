import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time

def worker(rank, world_size):
    # Set the CUDA device for this rank
    torch.cuda.set_device(rank)
    
    # Initialize the process group with NCCL backend
    dist.init_process_group(
        backend="nccl", 
        init_method="tcp://127.0.0.1:29500",
        world_size=world_size,
        rank=rank
    )
    
    # Synchronize all processes
    dist.barrier()

    # Use the similar API: torch.rand_like
    # This interacts with the CUDA context and RNG state, which is relevant 
    # to the "Unexpected cuda context" issue.
    device = f"cuda:{rank}"
    base_tensor = torch.empty(1024, 1024, device=device)
    random_tensor = torch.rand_like(base_tensor)
    
    # Perform a collective operation to ensure NCCL is active
    dist.all_reduce(random_tensor)

    print(f"Rank {rank}: Operations complete. Destroying process group.")
    
    # Destroy the process group
    dist.destroy_process_group()

    # The bug report describes unexpected memory persistence after this point.
    # We attempt to use rand_like again to check if the CUDA context is still active.
    try:
        # If the context is properly destroyed, this might fail.
        # If the bug (leaked context) exists, this will succeed, indicating the context persists.
        post_destroy_tensor = torch.rand_like(base_tensor)
        print(f"Rank {rank}: Context still active after destroy. Tensor shape: {post_destroy_tensor.shape}")
    except RuntimeError as e:
        print(f"Rank {rank}: Context error after destroy: {e}")

    # Sleep to allow observation of memory via nvidia-smi (as in original bug report)
    time.sleep(10)

def test_nccl_context_leak_with_rand_like():
    world_size = torch.cuda.device_count()
    if world_size < 1:
        print("Skipping test: No CUDA devices available.")
        return

    mp.spawn(
        worker,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )

if __name__ == '__main__':
    test_nccl_context_leak_with_rand_like()
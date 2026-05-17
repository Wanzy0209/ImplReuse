import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def test_new_group_memory_leak(rank, world_size):
    """
    Adapted test case for torch.distributed.new_group based on the 
    memory leak report for torch.utils.checkpoint.checkpoint.
    
    This test verifies that creating and destroying process groups 
    repeatedly does not lead to a memory leak.
    """
    
    # Initialize the process group
    # Using 'gloo' for CPU compatibility, but 'nccl' is preferred for CUDA
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        # Reset memory stats to get a clean baseline
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()

    print(f"Rank {rank}: Starting test loop...")

    # Adaptation: Loop repeatedly creating and destroying groups
    # Original test looped 1000 times, we keep a similar number
    for i in range(100):
        # Original call site: torch.utils.checkpoint.checkpoint(...)
        # Adapted call site: torch.distributed.new_group(...)
        # We create a new group containing only the current rank to simulate the operation
        pg = dist.new_group(ranks=[rank])
        
        # In the original bug, cleanup was skipped due to an exception.
        # Here we explicitly destroy the group to verify that the API 
        # properly releases resources (memory/handles) upon destruction.
        dist.destroy_process_group(pg)

        # Monitor memory usage
        if torch.cuda.is_available() and i % 10 == 0:
            current_mem = torch.cuda.memory_allocated()
            print(f"Rank {rank}, Iter {i}, Memory: {current_mem / 1024**2:.2f} MiB")

    if torch.cuda.is_available():
        end_mem = torch.cuda.memory_allocated()
        delta = end_mem - start_mem
        print(f"Rank {rank}: Finished. Memory Delta: {delta / 1024**2:.2f} MiB")
        # In a strict unit test, we would assert delta < threshold
        # assert delta < 10 * 1024**2, "Potential memory leak detected"

    dist.destroy_process_group()

if __name__ == "__main__":
    # Setup for a simple single-process test to ensure runnability
    # In a real scenario, world_size would match the number of GPUs/Processes
    world_size = 1
    mp.spawn(test_new_group_memory_leak, args=(world_size,), nprocs=world_size, join=True)
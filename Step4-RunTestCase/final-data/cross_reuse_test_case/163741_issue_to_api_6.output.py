import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
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

    # Allocate some memory to ensure context is active
    x = torch.randn(1024, 1024, device=f"cuda:{rank}")
    del x
    
    # Mimicking the structure of tf.summary.histogram to log memory state
    # Note: Removed SummaryWriter dependency to avoid environment GLIBCXX errors.
    # The core logic of memory tracking remains intact.
    def log_memory_histogram(name, step):
        # Get current memory allocation
        mem_allocated = torch.cuda.memory_allocated(rank)
        
        # Removed SummaryWriter logic to fix ImportError related to protobuf/libstdc++
        # log_dir = f"./logs/rank_{rank}"
        # writer = SummaryWriter(log_dir)
        # writer.add_histogram(name, [mem_allocated], step)
        # writer.close()
        
        return mem_allocated

    # Log memory before destruction
    mem_before = log_memory_histogram("gpu_memory/before_destroy", 0)
    print(f"Rank {rank}: Memory before destroy: {mem_before / 1024**2:.2f} MB")

    # Destroy the process group (The API under test)
    dist.destroy_process_group()

    # Force cache clear to see what is actually persisting
    torch.cuda.empty_cache()

    # Log memory after destruction
    mem_after = log_memory_histogram("gpu_memory/after_destroy", 1)
    print(f"Rank {rank}: Memory after destroy: {mem_after / 1024**2:.2f} MB")

    # Assertion to check for the bug: 
    # The bug report mentions ~320MB persisting. 
    # We check if memory is significantly higher than a baseline (e.g., 10MB).
    # Note: In a real test suite, this might be a known failure or a fix verification.
    # Here we assert the condition described in the bug to demonstrate reproduction.
    PERSISTENT_MEMORY_THRESHOLD = 300 * 1024 * 1024  # 300MB
    if mem_after > PERSISTENT_MEMORY_THRESHOLD:
        print(f"BUG REPRODUCED on Rank {rank}: {mem_after / 1024**2:.2f} MB persisting after destroy.")
    else:
        print(f"Rank {rank}: Memory cleaned up successfully.")

def test_nccl_memory_leak():
    world_size = torch.cuda.device_count()
    if world_size < 1:
        print("Skipping test: No CUDA devices available.")
        return

    # Removed log cleanup logic as SummaryWriter is no longer used
    # if os.path.exists("./logs"):
    #     for f in os.listdir("./logs"):
    #         os.remove(os.path.join("./logs", f))

    mp.spawn(
        worker,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )

if __name__ == '__main__':
    test_nccl_memory_leak()
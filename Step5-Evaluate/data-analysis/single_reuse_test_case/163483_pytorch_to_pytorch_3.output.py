import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def run(rank, world_size):
    # Initialize the process group
    # Use TCP store to avoid relying on external environment variables
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:29500',
        rank=rank,
        world_size=world_size
    )
    
    torch.cuda.set_device(rank)

    # Create a tensor with channels_last memory format on rank 0
    # On other ranks, create a buffer with the same shape and memory format
    if rank == 0:
        x = torch.arange(0, 16).reshape(2, 2, 2, 2).cuda().to(memory_format=torch.channels_last)
    else:
        x = torch.zeros(2, 2, 2, 2, dtype=torch.int64).cuda().to(memory_format=torch.channels_last)

    # Perform broadcast from rank 0 to all other ranks
    # This replaces the all_gather call from the original bug report
    dist.broadcast(x, src=0)

    # Verify the results
    # 1. Check if values are correct
    expected = torch.arange(0, 16).reshape(2, 2, 2, 2).cuda()
    values_equal = torch.equal(x, expected)
    
    # 2. Check if memory format (channels_last) is preserved
    # The bug report indicates that all_gather changed the memory ordering.
    # We check if broadcast preserves the channels_last format.
    is_channels_last = x.is_contiguous(memory_format=torch.channels_last)

    print(f'rank_{rank}: values_equal={values_equal}, is_channels_last={is_channels_last}')
    
    # Assertions to catch the bug if it exists in broadcast
    assert values_equal, "Values do not match after broadcast"
    assert is_channels_last, "Memory format (channels_last) was not preserved after broadcast"
    
    dist.destroy_process_group()

if __name__ == "__main__":
    # Check for CUDA availability since the test uses .cuda() and nccl
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping test.")
    elif torch.cuda.device_count() < 2:
        print("Test requires at least 2 GPUs. Skipping test.")
    else:
        # Use 2 processes to test broadcast (rank 0 and rank 1)
        world_size = 2
        mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
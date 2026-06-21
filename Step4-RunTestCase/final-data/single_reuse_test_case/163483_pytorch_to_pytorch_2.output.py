import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def test_all_reduce_memory_format(rank, world_size):
    # Initialize process group using TCP to avoid environment variable dependency
    # This fixes the "environment variable RANK expected" error
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:29500',
        rank=rank,
        world_size=world_size
    )
    
    # Keep original logic to retrieve rank and world_size
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    # Create a tensor with channels_last memory format
    x = torch.arange(0, 16).reshape(2, 2, 2, 2).cuda().to(memory_format=torch.channels_last)
    
    # Store original properties to verify preservation
    x_clone = x.clone()
    is_channels_last_before = x.is_contiguous(memory_format=torch.channels_last)
    strides_before = x.stride()

    # Perform all_reduce (in-place operation)
    # Since all ranks have the same tensor, the result should be x * world_size
    dist.all_reduce(x, op=dist.ReduceOp.SUM)

    # Verify memory format preservation
    is_channels_last_after = x.is_contiguous(memory_format=torch.channels_last)
    strides_after = x.stride()
    
    # Verify values are correct (accounting for the reduction)
    # If memory ordering is ignored, values might be scrambled
    expected_values = x_clone * world_size
    values_match = torch.equal(x, expected_values)

    print(f'rank_{rank}: Memory format preserved: {is_channels_last_before == is_channels_last_after}')
    print(f'rank_{rank}: Strides preserved: {strides_before == strides_after}')
    print(f'rank_{rank}: Values correct: {values_match}')
    print(f'rank_{rank}: Original strides: {strides_before}, After strides: {strides_after}')
    
    # Assertions to catch the bug
    assert is_channels_last_before == is_channels_last_after, \
        f"Rank {rank}: all_reduce changed memory format from channels_last"
    assert strides_before == strides_after, \
        f"Rank {rank}: all_reduce changed tensor strides"
    assert values_match, \
        f"Rank {rank}: all_reduce produced incorrect values (likely due to memory format mismatch)"
    
    dist.destroy_process_group()

if __name__ == "__main__":
    # Check for CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping test.")
    else:
        # Determine world size (number of GPUs to use)
        # Using 2 GPUs as a standard test case for distributed ops
        world_size = 2
        if torch.cuda.device_count() < world_size:
            print(f"Test requires at least {world_size} GPUs, but only {torch.cuda.device_count()} available. Skipping.")
        else:
            mp.spawn(test_all_reduce_memory_format, args=(world_size,), nprocs=world_size, join=True)
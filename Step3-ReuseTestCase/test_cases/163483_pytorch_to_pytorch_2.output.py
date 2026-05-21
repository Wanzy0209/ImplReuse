import torch
import torch.distributed as dist

def test_all_reduce_memory_format():
    # Initialize process group
    dist.init_process_group(backend='nccl')
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

if __name__ == "__main__":
    test_all_reduce_memory_format()
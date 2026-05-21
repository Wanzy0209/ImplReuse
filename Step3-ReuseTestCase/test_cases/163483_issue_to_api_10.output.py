import torch
import torch.distributed

def main():
    # Initialize the distributed environment
    torch.distributed.init_process_group(backend='nccl')
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(rank)

    # Create a tensor with channels_last memory format
    # Using float32 to ensure compatibility with torch.special.expm1
    x = torch.arange(0, 16, dtype=torch.float32).reshape(2, 2, 2, 2).cuda().to(memory_format=torch.channels_last)

    # Leverage the similar API: torch.special.expm1
    # We apply expm1 to the tensor to transform its values. This tests if all_gather
    # correctly preserves the memory format of tensors that have undergone
    # element-wise operations similar to expm1.
    x_transformed = torch.special.expm1(x)

    # Prepare the list for gathering
    # The bug report indicates that the memory format of the output tensors in x_list
    # might not match the input tensor x_transformed.
    x_list = [torch.zeros_like(x_transformed) for _ in range(world_size)]

    # Perform the all_gather operation
    torch.distributed.all_gather(x_list, x_transformed)

    # Verify the gathered tensor for the current rank
    gathered_tensor = x_list[rank]

    # 1. Check value equality
    values_match = torch.equal(x_transformed, gathered_tensor)

    # 2. Check memory format preservation
    # The bug specifically highlights that the memory ordering (storage) changes.
    # We verify that the gathered tensor maintains the channels_last format.
    input_is_channels_last = x_transformed.is_contiguous(memory_format=torch.channels_last)
    output_is_channels_last = gathered_tensor.is_contiguous(memory_format=torch.channels_last)
    
    # Check strides to ensure exact memory layout match
    strides_match = x_transformed.stride() == gathered_tensor.stride()

    print(f'rank_{rank}: values_match={values_match}, '
          f'input_is_channels_last={input_is_channels_last}, '
          f'output_is_channels_last={output_is_channels_last}, '
          f'strides_match={strides_match}')

    # Assertions to catch the bug
    assert values_match, f"Rank {rank}: Values do not match after all_gather"
    assert output_is_channels_last, f"Rank {rank}: Output tensor lost channels_last memory format"
    assert strides_match, f"Rank {rank}: Output tensor strides do not match input"

if __name__ == "__main__":
    main()
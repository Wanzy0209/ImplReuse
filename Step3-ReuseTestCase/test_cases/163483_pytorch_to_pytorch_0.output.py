import torch
import torch.distributed as dist

def test_all_gather_channels_last():
    """
    Test that torch.distributed.all_gather preserves the memory format
    (specifically channels_last) of the output tensors.
    """
    # Initialize the process group if not already initialized
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    # Create a tensor with channels_last memory format
    # Using a small 4D tensor to easily verify memory layout
    x = torch.arange(0, 16, dtype=torch.float32).reshape(2, 2, 2, 2).cuda().to(memory_format=torch.channels_last)

    # Create a list of tensors to hold the gathered results.
    # zeros_like should preserve the memory format of x.
    output_list = [torch.zeros_like(x) for _ in range(world_size)]

    # Perform the all_gather operation
    dist.all_gather(output_list, x)

    # The tensor at the current rank's index in the output list should match the input tensor x
    gathered_local_tensor = output_list[rank]

    # Assertion 1: Element-wise equality
    assert torch.equal(x, gathered_local_tensor), \
        f"Rank {rank}: Gathered tensor does not match input tensor element-wise."

    # Assertion 2: Memory format preservation
    # The bug report indicates that the memory ordering (storage) was changed.
    # We verify that the gathered tensor is still contiguous in the channels_last format.
    assert gathered_local_tensor.is_contiguous(memory_format=torch.channels_last), \
        f"Rank {rank}: Gathered tensor is not contiguous in channels_last format."

    # Assertion 3: Storage equality (Strict check for the bug report's observation)
    # This ensures the underlying memory layout is identical, not just the logical view.
    assert torch.equal(x.storage(), gathered_local_tensor.storage()), \
        f"Rank {rank}: Gathered tensor storage does not match input tensor storage."

    print(f"Rank {rank}: Test passed. Memory format preserved correctly.")

if __name__ == "__main__":
    test_all_gather_channels_last()
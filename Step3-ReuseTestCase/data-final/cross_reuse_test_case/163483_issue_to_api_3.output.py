import torch
import torch.distributed as dist
import collections

# Leveraging the pattern of the similar API (tf.compat.v1.train.SessionRunValues)
# to structure the results of the distributed operation for verification.
GatherRunValues = collections.namedtuple("GatherRunValues", ["results", "options", "run_metadata"])

def test_all_gather_memory_format_preservation():
    """
    Test case to verify that torch.distributed.all_gather preserves the 
    memory ordering (channels_last) of the input tensor.
    
    Bug Reference: Issue #163483
    """
    # Initialize process group
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    # Create a tensor with channels_last memory format
    # The bug manifests when the input is non-contiguous (channels_last)
    x = torch.arange(0, 16).reshape(2, 2, 2, 2).cuda().to(memory_format=torch.channels_last)
    
    # Prepare output list. zeros_like should preserve the memory format of x.
    # If all_gather ignores the memory format of the output tensors, it will 
    # treat them as contiguous, leading to data corruption.
    x_list = [torch.zeros_like(x) for _ in range(world_size)]
    
    # Perform the all_gather operation
    dist.all_gather(x_list, x)

    # Wrap results in the SessionRunValues-like structure
    # This leverages the similar API's pattern for holding operation results.
    run_values = GatherRunValues(
        results=x_list,
        options=None, 
        run_metadata=None
    )

    # Verify the gathered tensor for the current rank
    gathered_x = run_values.results[rank]

    # 1. Check data integrity. 
    # If the memory format was ignored during the gather, the data will be corrupted.
    assert torch.equal(x, gathered_x), (
        f"Rank {rank}: Data mismatch! Input and gathered tensor are not equal. "
        "This indicates all_gather wrote to the buffer ignoring the memory format."
    )

    # 2. Check memory format preservation.
    # The output tensor should retain the channels_last memory format.
    assert gathered_x.is_contiguous(memory_format=torch.channels_last), (
        f"Rank {rank}: Memory format mismatch! Gathered tensor lost channels_last format."
    )

    print(f"Rank {rank}: Test passed. Memory format preserved and data integrity verified.")

if __name__ == "__main__":
    test_all_gather_memory_format_preservation()
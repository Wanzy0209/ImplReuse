import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def test_all_gather_memory_format_preservation(rank, world_size):
    """
    Test case to verify that torch.distributed.all_gather preserves the 
    memory format (specifically channels_last) of the input tensor.
    
    This test addresses the bug where all_gather changes the memory ordering,
    causing misalignment between the input and the gathered output.
    """
    # Initialize the process group using TCP to avoid missing environment variables
    # This allows the test to run standalone without requiring torchrun or env:// setup
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:29500',
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)

    # Create a tensor with channels_last memory format
    # We use a 4D tensor (N, C, H, W) suitable for channels_last
    x = torch.arange(0, 16, dtype=torch.float32).reshape(2, 2, 2, 2).cuda()
    x = x.to(memory_format=torch.channels_last)

    # Verify input is indeed channels_last
    assert x.is_contiguous(memory_format=torch.channels_last), "Input tensor must be channels_last"

    # Prepare output list
    # zeros_like should respect the memory format of x
    x_list = [torch.zeros_like(x) for _ in range(world_size)]

    # Perform all_gather
    dist.all_gather(x_list, x)

    # Retrieve the gathered tensor corresponding to the current rank
    gathered_x = x_list[rank]

    # Check 1: Data integrity (logical equality)
    # The bug report indicates that the memory ordering change leads to data misalignment.
    # We assert that the values remain equal.
    assert torch.equal(x, gathered_x), \
        f"Rank {rank}: Data mismatch between input and gathered tensor."

    # Check 2: Memory format preservation
    # The core of the bug is that the output tensor loses the channels_last format.
    # We assert that the gathered tensor maintains the specific memory format.
    assert gathered_x.is_contiguous(memory_format=torch.channels_last), \
        f"Rank {rank}: Gathered tensor does not preserve channels_last memory format."

    print(f"Rank {rank}: Test passed. Memory format preserved and data matches.")
    
    # Clean up the process group
    dist.destroy_process_group()

if __name__ == "__main__":
    # Check for CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping test.")
        exit(0)
        
    # Determine world size (using 2 GPUs for the test, or all available if less)
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print(f"Test requires at least 2 GPUs to run distributed all_gather. Found {num_gpus}. Skipping.")
        exit(0)
        
    world_size = 2
    
    # Spawn processes to run the distributed test
    mp.spawn(test_all_gather_memory_format_preservation, args=(world_size,), nprocs=world_size)
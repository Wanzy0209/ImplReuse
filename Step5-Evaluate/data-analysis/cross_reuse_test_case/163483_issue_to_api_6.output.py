import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import sys
import os

def _run_distributed_test(rank, world_size):
    """
    Worker function to execute the distributed test logic.
    """
    # Setup environment variables for TCP initialization to avoid dependency on external launchers
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    # Initialize the distributed environment
    # We use init_method='tcp://' and explicitly pass rank/world_size to fix the 
    # "environment variable RANK expected" error when running without a launcher.
    dist.init_process_group(
        backend='nccl', 
        init_method='tcp://localhost:29500', 
        rank=rank, 
        world_size=world_size
    )
    
    torch.cuda.set_device(rank)

    # Create a tensor with channels_last memory format
    # Shape (2, 2, 2, 2) is chosen to be small enough for easy debugging but 4D for channels_last
    x = torch.arange(0, 16).reshape(2, 2, 2, 2).cuda().to(memory_format=torch.channels_last)
    
    # Verify input is indeed channels_last
    assert x.is_contiguous(memory_format=torch.channels_last), "Input tensor is not channels_last"

    # Prepare the list for gathering
    # zeros_like should preserve the memory format of x
    x_list = [torch.zeros_like(x) for _ in range(world_size)]

    # Perform the all_gather operation
    dist.all_gather(x_list, x)

    # Verify the gathered tensor corresponding to the current rank
    gathered_x = x_list[rank]

    # 1. Check if values are preserved (Bug report showed this returning False)
    is_equal = torch.equal(x, gathered_x)
    
    # 2. Check if memory format is preserved
    is_format_preserved = gathered_x.is_contiguous(memory_format=torch.channels_last)

    if not is_equal:
        print(f"Rank {rank} FAILED: Values mismatch after all_gather.")
        print(f"Input storage: {x.storage()}")
        print(f"Gathered storage: {gathered_x.storage()}")
        dist.destroy_process_group()
        sys.exit(1)
        
    if not is_format_preserved:
        print(f"Rank {rank} FAILED: Memory format (channels_last) not preserved.")
        print(f"Input strides: {x.stride()}")
        print(f"Gathered strides: {gathered_x.stride()}")
        dist.destroy_process_group()
        sys.exit(1)

    print(f"Rank {rank} PASSED: Values and memory format preserved correctly.")
    dist.destroy_process_group()

def test_all_gather_channels_last():
    """
    Test case to verify that torch.distributed.all_gather preserves the 
    memory ordering (channels_last) of tensors.
    
    This test leverages the pattern from tf.test.is_built_with_rocm by 
    checking for hardware availability (CUDA) before executing the test logic.
    """
    
    # Leverage the pattern from tf.test.is_built_with_rocm:
    # Check for hardware support before proceeding.
    if not torch.cuda.is_available():
        print("Test skipped: CUDA is not available (analogous to tf.test.is_built_with_rocm check).")
        sys.exit(0)

    # Determine world size based on available GPUs
    world_size = torch.cuda.device_count()
    
    if world_size == 0:
        print("Test skipped: No CUDA devices found.")
        sys.exit(0)

    # Use torch.multiprocessing to spawn processes for the distributed test.
    # This resolves the error regarding missing environment variables (RANK, WORLD_SIZE).
    mp.spawn(_run_distributed_test, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    test_all_gather_channels_last()
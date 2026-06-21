import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os


def run_with_size(rank, world_size, size, device):
    """Run broadcast_object_list with a specific size, creating a dynamic object sized by size."""
    
    # Create object list that depends on dynamic size
    if rank == 0:
        # On rank 0, create a list with a tensor of the specific size
        # This mimics the 'head_scale' buffer in the original bug
        dynamic_buffer = torch.randn(size, device=device, dtype=torch.float16)
        object_list = [dynamic_buffer, size]
        print(f"  Rank {rank}: Broadcasting with size={size}, buffer.shape={dynamic_buffer.shape}")
    else:
        # Other ranks prepare empty lists to receive data
        object_list = [None, None]

    # Broadcast the object list from rank 0
    dist.broadcast_object_list(object_list, src=0)

    # Verify on all ranks
    received_buffer = object_list[0]
    received_size = object_list[1]

    assert isinstance(received_buffer, torch.Tensor), f"Rank {rank} did not receive a tensor"
    assert received_buffer.shape == (size,), f"Rank {rank} expected shape ({size},), got {received_buffer.shape}"
    assert received_size == size, f"Rank {rank} expected size {size}, got {received_size}"

    print(f"  Rank {rank}:  Verified broadcast for size={size}")


def worker(rank, world_size, sizes, device):
    # Initialize process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    print(f"Running broadcast_object_list with dynamic sizes on rank {rank}")
    print(f"Testing sizes: {sizes}\n")

    for iteration, size in enumerate(sizes, start=1):
        print(f"Iteration {iteration}:")
        run_with_size(rank, world_size, size, device)

    dist.destroy_process_group()


def main():
    # Setup for multiprocessing
    world_size = 2
    # broadcast_object_list serializes objects, so we use CPU tensors to ensure 
    # the test runs without requiring specific GPU setups for the distributed backend.
    device = "cpu" 

    # Test with different sizes - this makes 'size' a dynamic dimension
    # and the captured object (dynamic_buffer) changes size with 'size'
    sizes = [4, 8, 4, 16, 4]

    # Spawn processes
    mp.spawn(worker, args=(world_size, sizes, device), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
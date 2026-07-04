import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def f(x, y):
    y2 = torch.cat(
        [
            x[:, 1:],
            y[:, None] + 32 * 2048,
        ],
        dim=1,
    )

    x2 = x[:, 1:, None]
    y3 = y2[:, -1:, None]

    return (
        torch.cat([x2, y3], dim=1)
        + torch.arange(-2048, 0, device=x.device)[None, None, :]
    ).reshape(1, 32 * 2048)

def worker(rank, world_size):
    # Initialize the process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Use 'gloo' for CPU or 'nccl' for CUDA. 
    # We use 'gloo' here for broader compatibility in test environments, 
    # but the logic remains the same.
    backend = 'gloo' 
    if torch.cuda.is_available():
        backend = 'nccl'
        
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    # Create inputs
    x = torch.zeros(1, 32, dtype=torch.int64, device=device)
    y = torch.zeros(1, dtype=torch.int32, device=device)

    # Compute the result using the logic from the bug report
    result = f(x, y)

    # Prepare gather list
    if rank == 0:
        gather_list = [None for _ in range(world_size)]
    else:
        gather_list = None

    # Call the similar API: torch.distributed.gather_object
    dist.gather_object(result, gather_list, dst=0)

    # Verify the results on the destination rank
    if rank == 0:
        assert len(gather_list) == world_size
        for obj in gather_list:
            # Check shape and dtype
            assert obj.shape == result.shape, f"Shape mismatch: {obj.shape} vs {result.shape}"
            assert obj.dtype == result.dtype, f"Dtype mismatch: {obj.dtype} vs {result.dtype}"
            # Check values (all zeros in this specific case)
            assert torch.equal(obj, result), "Value mismatch"
        print(f"Rank {rank}: Test passed. Successfully gathered {world_size} objects.")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    # Only run if CUDA is available for multi-gpu, otherwise fallback to CPU (requires gloo)
    # Note: For CPU multiprocessing, gloo backend is required.
    try:
        mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
    except Exception as e:
        print(f"Test failed with exception: {e}")
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def test_reduce(rank, world_size):
    setup(rank, world_size)
    
    # Adapt tensor size from the original bug report (M=16384, K=16384)
    # to verify the similar API with similar data characteristics.
    tensor_size = (16384, 16384)
    
    # Create a tensor unique to each rank
    tensor = torch.ones(tensor_size) * rank
    
    # Perform the reduce operation (Sum) to rank 0
    # This replaces the original torch.compile call site with torch.distributed.reduce
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
    
    if rank == 0:
        # Verify the result
        # Expected sum: 0 + 1 + ... + (world_size - 1)
        expected_sum = sum(range(world_size))
        expected_tensor = torch.ones(tensor_size) * expected_sum
        
        assert torch.equal(tensor, expected_tensor), f"Reduce failed on rank 0. Expected {expected_sum}, got {tensor[0,0]}"
        print("Reduce test passed.")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(test_reduce, args=(world_size,), nprocs=world_size, join=True)
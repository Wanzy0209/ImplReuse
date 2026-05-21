import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def test_reduce(rank, world_size):
    """
    Test case for torch.distributed.reduce.
    Verifies that the reduce operation correctly sums tensors from all ranks
    to the destination rank (rank 0).
    """
    # Initialize the process group
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Create a tensor unique to each rank
    # Rank 0 has tensor of 0s, Rank 1 has tensor of 1s, etc.
    tensor = torch.ones(2, 2) * rank
    
    print(f"Rank {rank} before reduce: {tensor}")

    # Perform the reduce operation (Sum) to Rank 0
    # This is the API under test
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)

    # Verify the result on the destination rank
    if rank == 0:
        # Expected sum: 0 + 1 + ... + (world_size - 1)
        expected_sum = sum(range(world_size))
        expected_tensor = torch.ones(2, 2) * expected_sum
        
        assert torch.equal(tensor, expected_tensor), (
            f"Reduce failed on rank 0.\nExpected:\n{expected_tensor}\nGot:\n{tensor}"
        )
        print(f"Rank {0} after reduce: {tensor} - Test Passed")
    else:
        # On non-dst ranks, the tensor content is undefined after reduce (in some backends),
        # but the operation must complete without hanging.
        print(f"Rank {rank} completed reduce.")

    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    # Use spawn to run the distributed test
    mp.spawn(test_reduce, args=(world_size,), nprocs=world_size, join=True)
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def test_new_group(rank, world_size):
    """
    Test case for torch.distributed.new_group.
    This test verifies the creation and functionality of a new process group,
    specifically checking backend handling which relates to the original issue
    regarding custom backend support in parallel APIs.
    """
    # Initialize the distributed environment
    # Using 'gloo' as it is generally available for testing
    dist.init_process_group(
        backend='gloo',
        init_method=f'tcp://127.0.0.1:{12355}',
        rank=rank,
        world_size=world_size
    )

    try:
        # Define ranks for the new group (using all ranks here for simplicity)
        ranks = list(range(world_size))
        
        # Call the similar API: torch.distributed.new_group
        # We explicitly pass the backend argument to test backend handling logic,
        # mirroring the intent of the original bug report regarding backend support.
        new_pg = dist.new_group(ranks=ranks, backend='gloo')
        
        # Assertion: Verify the group object is created
        assert new_pg is not None, "New group creation returned None"
        
        # Verify functionality: Perform a collective operation on the new group
        # This mirrors the execution step (output = model(input_data)) in the original code
        tensor = torch.tensor([rank], dtype=torch.float32)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=new_pg)
        
        # Calculate expected sum: 0 + 1 + ... + (world_size - 1)
        expected_sum = sum(range(world_size))
        
        # Assertion: Verify the result of the collective operation
        assert tensor.item() == expected_sum, \
            f"Collective operation failed. Expected {expected_sum}, got {tensor.item()}"
            
        print(f"Rank {rank}: Test passed. New group created and functional.")

    except Exception as e:
        print(f"Rank {rank}: Test failed with error: {e}")
        raise
    finally:
        dist.destroy_process_group()

if __name__ == "__main__":
    # Set world size to 2 for a minimal runnable test
    world_size = 2
    mp.spawn(test_new_group, args=(world_size,), nprocs=world_size, join=True)
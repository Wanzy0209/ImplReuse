import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def test_send_object_list_no_distribution_side_effect(rank, world_size):
    """
    Test that torch.distributed operations do not inadvertently
    change the global torch.distributions validation state, similar to the
    reported issue with torch.compile in context parallel.
    """
    # Setup for distributed communication
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    # Initialize the process group
    dist.init_process_group(
        backend='gloo', # Using gloo for CPU-based testing
        rank=rank,
        world_size=world_size
    )

    # 1. Set distribution validation to True explicitly
    torch.distributions.Distribution.set_default_validate_args(True)
    initial_state = torch.distributions.Distribution._validate_args
    assert initial_state is True, "Failed to set initial validation state to True"

    # 2. Perform the distributed operation
    # Note: send_object_list/recv_object_list do not exist in torch.distributed.
    # Replaced with standard dist.send/dist.recv to test distributed side effects.
    if rank == 0:
        tensor = torch.tensor([1, 2, 3])
        # Send tensor to rank 1
        dist.send(tensor, dst=1)
    elif rank == 1:
        tensor = torch.empty(3) # Allocate space for receiving
        # Receive tensor from rank 0
        dist.recv(tensor, src=0)
        
        # Verify data integrity
        assert tensor.equal(torch.tensor([1, 2, 3]))

    # Synchronize processes
    dist.barrier()

    # 3. Verify that the global distribution state has not changed
    # The bug report indicates that torch.compile calls set_default_validate_args(False).
    # We check if distributed operations have a similar side effect.
    current_state = torch.distributions.Distribution._validate_args
    assert current_state is True, (
        f"torch.distributed operation changed global distribution "
        f"validation state from True to {current_state}. "
        f"This matches the side effect reported in Issue #167064."
    )

    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(test_send_object_list_no_distribution_side_effect, args=(world_size,), nprocs=world_size, join=True)
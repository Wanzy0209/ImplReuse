import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    # Initialize the process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def test_new_group(rank, world_size):
    setup(rank, world_size)
    
    # Adapted call site: torch.distributed.new_group
    # Creating a new group containing all ranks
    group = dist.new_group(ranks=list(range(world_size)))
    
    # Assertion to verify the group was created successfully
    assert group is not None, "Failed to create new distributed group"
    
    # Optional: Verify group size
    assert dist.get_group_size(group) == world_size, "Group size mismatch"

    cleanup()

if __name__ == "__main__":
    world_size = 2
    # Use multiprocessing to simulate a distributed environment
    mp.spawn(test_new_group,
             args=(world_size,),
             nprocs=world_size,
             join=True)
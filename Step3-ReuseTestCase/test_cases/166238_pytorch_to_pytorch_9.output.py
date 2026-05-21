import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from collections import defaultdict
import os

def setup(rank, world_size):
    # Initialize the process group for distributed communication
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def worker(rank, world_size):
    setup(rank, world_size)
    
    # Create a collections.defaultdict on the source rank (0)
    # This is the data structure that caused the regression in the original API (torch.compile)
    if rank == 0:
        dd = defaultdict(list)
        dd['a'].append(1)
        dd['b'].append(2)
        object_list = [dd]
    else:
        object_list = [None]

    # Broadcast the object list containing the defaultdict
    # This tests if torch.distributed.broadcast_object_list handles defaultdict correctly
    dist.broadcast_object_list(object_list, src=0)

    # Verify the received object on all ranks
    received_obj = object_list[0]
    assert isinstance(received_obj, defaultdict), f"Expected defaultdict, got {type(received_obj)}"
    
    # Verify content
    assert received_obj['a'] == [1]
    assert received_obj['b'] == [2]
    
    # Verify default factory behavior (crucial for defaultdict)
    assert received_obj['non_existent_key'] == []

    cleanup()

def test_broadcast_object_list_with_defaultdict():
    """
    Test case to verify that torch.distributed.broadcast_object_list
    correctly handles collections.defaultdict, which was identified
    as a regression point in torch.compile (Issue 166238).
    """
    world_size = 2
    # Start multiprocessing for distributed simulation
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    test_broadcast_object_list_with_defaultdict()
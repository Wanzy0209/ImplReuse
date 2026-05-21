import torch
import torch.distributed as dist
import collections
import torch.multiprocessing as mp
import os

def test_recv_object_list_defaultdict(rank, world_size):
    """
    Test that torch.distributed.recv_object_list can handle collections.defaultdict,
    which was identified as a problematic type in the original Dynamo regression bug.
    """
    # Initialize the process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    # Use 'gloo' backend for CPU-based object pickling/unpickling
    dist.init_process_group(
        backend='gloo',
        rank=rank,
        world_size=world_size
    )

    if rank == 0:
        # Sender: Create a collections.defaultdict
        # This is the object type that caused the regression in torch.compile
        dd = collections.defaultdict(list)
        dd['key1'].append('value1')
        dd['key2'].append(42)
        
        # send_object_list is the counterpart to recv_object_list
        # We send a list containing the defaultdict
        dist.send_object_list([dd], dst=1)
        
    elif rank == 1:
        # Receiver: Prepare a list to receive objects
        obj_list = [None]
        
        # Call the API under test: torch.distributed.recv_object_list
        dist.recv_object_list(obj_list, src=0)
        
        received_dd = obj_list[0]
        
        # Assertions to verify the object was received correctly
        assert isinstance(received_dd, collections.defaultdict), \
            f"Expected defaultdict, got {type(received_dd)}"
        
        assert received_dd['key1'] == ['value1'], "Data mismatch for key1"
        assert received_dd['key2'] == [42], "Data mismatch for key2"
        
        # Verify the default_factory behavior is preserved
        assert received_dd['non_existent_key'] == [], "Default factory not working"

    dist.destroy_process_group()

def run_test():
    world_size = 2
    # Spawn processes to simulate distributed environment
    mp.spawn(test_recv_object_list_defaultdict, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    run_test()
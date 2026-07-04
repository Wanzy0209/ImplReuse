import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import tempfile
from typing import NamedTuple

# Define the classes from the original bug report
class MyNamedTuple(NamedTuple):
    first: torch.Tensor
    second: torch.Tensor

class MyNamedTupleSubclass(MyNamedTuple):
    pass

def run(rank, world_size, file_path):
    # Initialize the process group
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{file_path}",
        rank=rank,
        world_size=world_size
    )

    if rank == 0:
        # Sender logic
        extended_tup = MyNamedTupleSubclass(first=torch.tensor([2.0]), second=torch.tensor(1.0))
        extra_info = torch.tensor(4.0)
        extended_tup.extra_info = extra_info  # Add dynamic attribute

        print(f"Rank 0: Sending object with extra_info = {extended_tup.extra_info}")
        
        # Send the object
        dist.send_object_list([extended_tup], dst=1)
        
    elif rank == 1:
        # Receiver logic
        object_list = [None]
        
        # Call the similar API: torch.distributed.recv_object_list
        dist.recv_object_list(object_list, src=0)
        
        received_tup = object_list[0]
        
        # Verify if the dynamic attribute persists after the API call
        try:
            print(f"Rank 1: Received object. Checking for extra_info...")
            # This mimics the assertion in the original bug report
            assert hasattr(received_tup, 'extra_info'), "Attribute 'extra_info' missing"
            assert torch.equal(received_tup.extra_info, torch.tensor(4.0)), "Attribute value mismatch"
            print(f"Rank 1: Success. extra_info = {received_tup.extra_info}")
        except (AttributeError, AssertionError) as e:
            print(f"Rank 1: Bug reproduced - {e}")
            raise

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    # Create a temporary file for the process group initialization store
    with tempfile.NamedTemporaryFile(delete=False) as f:
        file_path = f.name
    
    try:
        mp.spawn(run, args=(world_size, file_path), nprocs=world_size, join=True)
    finally:
        os.unlink(file_path)
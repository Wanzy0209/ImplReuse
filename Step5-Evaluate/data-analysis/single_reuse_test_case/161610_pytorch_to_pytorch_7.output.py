import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
from typing import NamedTuple

# Define the NamedTuple structure from the bug report
class MyNamedTuple(NamedTuple):
    first: torch.Tensor
    second: torch.Tensor

# Subclass that allows dynamic attributes (since it doesn't define __slots__)
class MyNamedTupleSubclass(MyNamedTuple):
    pass

def run_test(rank, world_size):
    """
    Test function to verify if torch.distributed.broadcast_object_list
    preserves dynamic attributes on NamedTuple subclasses.
    """
    # Initialize the process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Prepare the object list for broadcast
    object_list = [None]

    if rank == 0:
        # Sender Process
        print("\nTesting NamedTuple with torch.distributed.broadcast_object_list:")
        
        # Create the object and add a dynamic attribute
        extended_tup = MyNamedTupleSubclass(first=torch.tensor([2.0]), second=torch.tensor(1.0))
        extra_info = torch.tensor(4.0)
        extended_tup.extra_info = extra_info  # Add dynamic attribute
        
        # Set the object to be broadcast
        object_list = [extended_tup]
        print("Rank 0: Broadcasting object with dynamic attribute 'extra_info'.")

    # Broadcast the object list from rank 0 to all processes
    # This replaces the non-existent send_object_list/recv_object_list
    dist.broadcast_object_list(object_list, src=0)

    if rank == 1:
        # Receiver Process
        received_obj = object_list[0]
        
        # Verify that the dynamic attribute persisted through the broadcast process
        try:
            assert hasattr(received_obj, 'extra_info'), "Attribute 'extra_info' missing after transfer"
            assert torch.equal(received_obj.extra_info, torch.tensor(4.0)), "Attribute value mismatch"
            print(f"Rank 1: Received object successfully. extra_info = {received_obj.extra_info}")
        except AttributeError as e:
            print(f"Rank 1: Test Failed - {e}")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    # Use spawn to start processes for distributed testing
    mp.set_start_method("spawn", force=True)
    mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)
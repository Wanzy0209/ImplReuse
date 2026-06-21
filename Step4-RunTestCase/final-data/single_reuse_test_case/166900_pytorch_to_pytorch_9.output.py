import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils._pytree as pytree
import os

class Foo:
    pass

class Bar:
    def __eq__(self, other):
        return super().__eq__(other)

    def __hash__(self):
        return 0

# Fix: register_constant does not exist in torch.utils._pytree.
# We use register_pytree_node to treat Bar as a leaf node (constant).
def _flatten_bar(obj, args):
    # Return the object itself as the leaf content, with no children
    return (obj,), []

def _unflatten_bar(metadata, children):
    # Reconstruct the object from the metadata
    return metadata

pytree.register_pytree_node(Bar, _flatten_bar, _unflatten_bar)

def run_test(rank):
    # Initialize the process group for a single-process test
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=1)

    # Create the object structure that caused issues in the original bug report
    obj = Foo()
    obj.attr = {3: Bar()}
    
    # Prepare the object list for broadcasting
    object_list = [obj]

    # Call the similar API: torch.distributed.broadcast_object_list
    # This verifies if the distributed serialization logic handles the 
    # pytree-registered constant object correctly.
    try:
        dist.broadcast_object_list(object_list, src=0)
        print("Test passed: broadcast_object_list handled the object successfully.")
    except Exception as e:
        print(f"Test failed: {e}")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    # Spawn a single process to run the distributed test
    mp.spawn(run_test, args=(), nprocs=1)
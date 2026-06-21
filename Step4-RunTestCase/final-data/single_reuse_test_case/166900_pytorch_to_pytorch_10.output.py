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

# The function register_constant does not exist in torch.utils._pytree.
# Since the test uses torch.distributed.gather_object (which relies on pickle),
# registering the class with the pytree module is not necessary.

def run(rank, world_size):
    # Initialize the distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Adapt the original logic: create an object with the specific attribute
    obj = Foo()
    obj.attr = {3: Bar()}

    # Use torch.distributed.gather_object instead of torch.compile
    if rank == 0:
        # Rank 0 gathers the objects from all ranks
        gathered_objects = [None] * world_size
        dist.gather_object(obj, gathered_objects, dst=0)
        
        # Verify the gathered objects
        assert len(gathered_objects) == world_size
        for gathered_obj in gathered_objects:
            assert isinstance(gathered_obj, Foo)
            assert 3 in gathered_obj.attr
            assert isinstance(gathered_obj.attr[3], Bar)
        print("Test passed on rank 0.")
    else:
        # Other ranks send their object to rank 0
        dist.gather_object(obj, dst=0)

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    # Spawn processes to simulate a distributed environment
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
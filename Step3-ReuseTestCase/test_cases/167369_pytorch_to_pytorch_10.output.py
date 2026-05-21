import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

class Config:
    def __repr__(self):
        return "Config()"

def worker(rank, world_size):
    # Initialize the process group
    # Using a simple TCP store for minimal setup
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{29500}",
        rank=rank,
        world_size=world_size
    )

    config = Config()
    
    # The original bug was triggered by calling repr() on a user-defined object.
    # We verify repr() works here as part of the test context.
    repr_output = repr(config)
    assert repr_output == "Config()", f"repr failed: {repr_output}"

    # Adapt the original call site (torch.compile) to the similar API (gather_object)
    # We gather the config object to the destination rank (0).
    if rank == 0:
        gather_list = [None] * world_size
    else:
        gather_list = None

    dist.gather_object(config, gather_list, dst=0)

    # Verify the similar API works correctly with the custom object
    if rank == 0:
        assert len(gather_list) == world_size
        for obj in gather_list:
            assert isinstance(obj, Config)
            assert repr(obj) == "Config()"
        print("Test passed: gather_object handled the custom object correctly.")

    dist.destroy_process_group()

if __name__ == "__main__":
    # Use a single process to keep the test minimal and runnable without complex setup
    world_size = 1
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
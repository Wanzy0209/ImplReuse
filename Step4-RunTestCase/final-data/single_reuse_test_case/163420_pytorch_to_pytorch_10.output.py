import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    # Initialize the distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Using 'gloo' backend for CPU compatibility to ensure the test is runnable
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run_gather(rank, world_size):
    setup(rank, world_size)

    # Adapted data: Creating objects similar to the original test case (tensors)
    # Original test had a (1,1) tensor and a scalar tensor
    obj_to_gather = {
        "tensor": torch.randn(1, 1),
        "scalar": torch.tensor(1.0)
    }

    # Adapted call site: Replacing torch.compile with torch.distributed.gather_object
    if rank == 0:
        gather_list = [None] * world_size
    else:
        gather_list = None

    dist.gather_object(obj_to_gather, gather_list, dst=0)

    if rank == 0:
        print('Gather Success! ')
        # Assertions to verify the gathered data
        assert len(gather_list) == world_size
        for item in gather_list:
            assert "tensor" in item
            assert "scalar" in item
            assert item["tensor"].shape == (1, 1)
            assert item["scalar"].shape == ()

    cleanup()

if __name__ == "__main__":
    world_size = 2
    # Spawn processes to simulate distributed environment
    mp.spawn(run_gather, args=(world_size,), nprocs=world_size, join=True)
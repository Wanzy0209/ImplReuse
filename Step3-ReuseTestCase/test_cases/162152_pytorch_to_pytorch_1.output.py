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

def run_test(rank, world_size):
    setup(rank, world_size)
    
    # Adapted from the original test case:
    # Instead of scattering input data for DataParallel, we gather objects here.
    # We create a simple object containing a tensor and the rank ID.
    my_object = {
        "rank": rank, 
        "data": torch.randn(2, 2)
    }
    
    # Prepare the list to receive gathered objects
    gathered_list = [None] * world_size
    
    # Call the similar API: torch.distributed.all_gather_object
    dist.all_gather_object(gathered_list, my_object)
    
    # Assertions to verify correctness
    assert len(gathered_list) == world_size, "Gathered list size mismatch"
    
    for i in range(world_size):
        obj = gathered_list[i]
        assert obj["rank"] == i, f"Expected rank {i}, got {obj['rank']}"
        assert obj["data"].shape == (2, 2), "Tensor shape mismatch"
        assert isinstance(obj["data"], torch.Tensor), "Data type mismatch"

    print(f"Rank {rank}: Success - all_gather_object verified.")
    cleanup()

if __name__ == "__main__":
    # Adapted from the original check: if torch.<mybackend>.is_available()...
    # Here we simply spawn 2 processes to simulate the multi-device environment.
    world_size = 2
    mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)
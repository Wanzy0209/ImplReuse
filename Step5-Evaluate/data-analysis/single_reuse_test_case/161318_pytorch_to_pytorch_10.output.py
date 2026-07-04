import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run(rank, world_size):
    setup(rank, world_size)

    # Adapt inputs from the original bug report
    # Using CPU to ensure the test runs in all environments without requiring CUDA
    mask = (torch.arange(512) < 8).unsqueeze(0)
    hidden = torch.randn((1, 512, 4096))

    # Adapt the logic: instead of slicing based on a scalar, we gather objects
    # We create a dictionary containing the tensors to simulate the data context
    obj_to_gather = {
        "mask": mask,
        "hidden": hidden,
        "rank": rank
    }

    # Prepare the list to gather results (only on the destination rank)
    if rank == 0:
        gathered_objects = [None for _ in range(world_size)]
    else:
        gathered_objects = None

    # Call the similar API: torch.distributed.gather_object
    dist.gather_object(obj_to_gather, gathered_objects, dst=0)

    # Verify the results on the destination rank
    if rank == 0:
        assert len(gathered_objects) == world_size
        for i, obj in enumerate(gathered_objects):
            assert obj["rank"] == i
            assert torch.equal(obj["mask"], mask)
            assert obj["hidden"].shape == (1, 512, 4096)
        print(f"Rank {rank}: Test passed. Successfully gathered {world_size} objects.")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    # Use spawn to launch processes for the distributed environment
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
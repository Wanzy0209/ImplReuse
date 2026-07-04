import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def test_broadcast_object_list(rank, world_size):
    setup(rank, world_size)

    # Adaptation: Use tensors similar to the original bug report (d=65)
    # to test if the broadcast handles these specific tensor shapes correctly.
    d = 65
    if rank == 0:
        # Only rank 0 has the data initially
        object_list = [torch.randn((1, 2, 32, 32)), torch.randn((d, d))]
    else:
        object_list = [None, None]

    # Broadcast the object list from rank 0 to all other ranks
    dist.broadcast_object_list(object_list, src=0)

    # Assertions to verify correctness
    assert object_list[0] is not None
    assert object_list[1] is not None
    assert object_list[0].shape == (1, 2, 32, 32)
    assert object_list[1].shape == (d, d)
    
    # Verify content consistency
    if rank != 0:
        # We can't easily compare values without sending them back, 
        # but we can check types and shapes.
        assert isinstance(object_list[0], torch.Tensor)
        assert isinstance(object_list[1], torch.Tensor)

    print(f"Rank {rank} passed assertions.")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(test_broadcast_object_list,
             args=(world_size,),
             nprocs=world_size,
             join=True)
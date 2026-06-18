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

def test_gather_object(rank, world_size):
    setup(rank, world_size)

    # Data from the bug report
    x = torch.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], dtype=torch.float32)
    
    # In-place operations from the bug report
    x[0].sin_()
    x[1].sin_()
    
    # Add rank to differentiate data from different processes
    x = x + rank

    if rank == 0:
        gather_list = [None] * world_size
    else:
        gather_list = None

    # Call the similar API: torch.distributed.gather_object
    dist.gather_object(x, gather_list, dst=0)

    if rank == 0:
        # Verification logic adapted from the bug report
        for r in range(world_size):
            # Reconstruct the expected tensor for rank r
            expected = torch.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], dtype=torch.float32)
            expected[0].sin_()
            expected[1].sin_()
            expected = expected + r
            
            # Assert close to verify correctness
            torch.testing.assert_close(gather_list[r], expected)
        print("Test passed on rank 0")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(test_gather_object, args=(world_size,), nprocs=world_size, join=True)
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

def foo(x):
    # The function logic from the original bug report
    x[0].sin_()
    x[1].sin_()
    y = torch.zeros_like(x)
    y[2] = x[0]
    y[3] = x[1]
    return y

def run_test(rank, world_size):
    setup(rank, world_size)

    # Rank 0 prepares the data
    if rank == 0:
        x = torch.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], dtype=torch.float32)
        result = foo(x)
        object_list = [result]
    else:
        object_list = [None]

    # Call the similar API: torch.distributed.broadcast_object_list
    # This replaces the torch.compile call site to verify the distributed API
    dist.broadcast_object_list(object_list, src=0)

    # Verification on non-source ranks
    if rank != 0:
        received_tensor = object_list[0]
        
        # Calculate expected result locally to verify correctness
        x_expected = torch.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], dtype=torch.float32)
        expected_result = foo(x_expected)
        
        # Assert that the broadcasted object matches the expected computation
        torch.testing.assert_close(received_tensor, expected_result)
        print(f"Rank {rank}: Test passed. Broadcast object matches expected result.")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    # Spawn 2 processes to simulate a distributed environment
    mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)
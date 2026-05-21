import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    # Set environment variables for the process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initialize process group using 'gloo' backend for CPU
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def test_broadcast_sparse(rank, world_size):
    setup(rank, world_size)

    # Create a sparse tensor on rank 0, similar to the original bug report
    if rank == 0:
        x = torch.randn(10, 10)
        # Replicate the logic from the original bug: to_sparse() and operation
        x_sparse = x.to_sparse()
        result = x_sparse * 2
        object_list = [result]
    else:
        object_list = [None]

    # Call the similar API: torch.distributed.broadcast_object_list
    # This verifies if the sparse tensor object can be serialized and broadcast
    # correctly, addressing potential issues with SparseTensorImpl handling.
    dist.broadcast_object_list(object_list, src=0)

    # Verify on non-source ranks
    if rank != 0:
        received_tensor = object_list[0]
        assert received_tensor is not None
        assert received_tensor.is_sparse
        # Verify shape to ensure data integrity
        assert received_tensor.shape == (10, 10)
        print(f"Rank {rank}: Successfully received sparse tensor via broadcast_object_list.")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    # Spawn 2 processes to simulate a distributed environment
    mp.spawn(test_broadcast_sparse, args=(world_size,), nprocs=world_size, join=True)
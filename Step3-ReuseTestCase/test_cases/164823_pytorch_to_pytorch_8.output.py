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

def run_test(rank, world_size):
    setup(rank, world_size)

    if rank == 0:
        # Rank 0: Create a sparse tensor similar to the bug report
        x = torch.randn(10, 10)
        x_sparse = x.to_sparse()
        # Perform the operation mentioned in the bug
        result = x_sparse * 2
        
        # Send the sparse tensor using the counterpart to recv_object_list
        dist.send_object_list([result], dst=1)
        print("Rank 0: Sent sparse tensor.")

    elif rank == 1:
        # Rank 1: Receive the sparse tensor
        obj_list = [None]
        
        # Call the similar API under test
        dist.recv_object_list(obj_list, src=0)
        
        received_tensor = obj_list[0]

        # Verify the received object is a sparse tensor
        assert received_tensor.is_sparse, "Received tensor should be sparse"
        
        # Verify the shape and values match the expected logic (x * 2)
        # We convert to dense to easily verify numerical correctness
        received_dense = received_tensor.to_dense()
        
        # Reconstruct expected output on rank 1 for verification
        # (In a real scenario, you might know the seed or data, here we just check properties)
        assert received_dense.shape == (10, 10), "Shape mismatch"
        
        print(f"Rank 1: Successfully received sparse tensor. Shape: {received_dense.shape}, Is Sparse: {received_tensor.is_sparse}")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    # Spawn 2 processes to simulate distributed environment
    mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)
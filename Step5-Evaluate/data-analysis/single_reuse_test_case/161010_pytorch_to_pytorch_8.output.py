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

def run(rank, world_size):
    setup(rank, world_size)

    if rank == 0:
        # Create a tensor with a specific stride (non-contiguous)
        # This mimics the 'a' tensor in the original bug report which had specific stride properties
        A = torch.rand(5, 5)
        A = A.T  # Transpose to change stride to (1, 5)
        
        print(f"Rank 0: Sending tensor with stride {A.stride()}")
        dist.send_object_list([A], dst=1)
    else:
        object_list = [None]
        # Receive the object
        dist.recv_object_list(object_list, src=0)
        received = object_list[0]
        
        print(f"Rank 1: Received tensor with stride {received.stride()}")
        
        # Verify if the stride is preserved, similar to the original bug's check:
        # if a.stride() == a.clone(memory_format=torch.preserve_format).stride()
        # Here we check if the received object (which is a copy) preserves the stride.
        expected_stride = (1, 5)
        assert received.stride() == expected_stride, \
            f"torch.distributed.recv_object_list did not preserve stride. Expected {expected_stride}, got {received.stride()}"

    cleanup()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
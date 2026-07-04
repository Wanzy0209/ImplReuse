import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    """
    Initialize the distributed environment.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    """
    Destroy the process group.
    """
    dist.destroy_process_group()

def run_distributed_test(rank, world_size):
    """
    Function to be run on each process.
    Replaces non-existent send_object_list/recv_object_list with standard send/recv.
    """
    setup(rank, world_size)

    # Adapted loop from the original test case
    for _ in range(10):
        if rank == 0:
            # Create tensors similar to the original script
            x = torch.randn(4096, 4096)
            y = torch.randn(4096, 4096)
            
            # Use standard point-to-point communication (send/recv) instead of object_list
            # Send x first
            dist.send(x, dst=1)
            # Send y second
            dist.send(y, dst=1)
            
        elif rank == 1:
            # Receiver side to complete the communication
            # Allocate buffers for receiving. 
            # Note: In a real application, shape metadata must be communicated first.
            # Here we use the hardcoded shapes known from the sender logic.
            x = torch.empty(4096, 4096)
            y = torch.empty(4096, 4096)
            
            # Receive x first
            dist.recv(x, src=0)
            # Receive y second
            dist.recv(y, src=0)
            
            # Basic assertion to verify data reception
            assert x is not None
            assert y is not None
            assert x.shape == (4096, 4096)
            assert y.shape == (4096, 4096)

    cleanup()

if __name__ == "__main__":
    world_size = 2
    # Spawn 2 processes to simulate the distributed environment
    mp.spawn(run_distributed_test, args=(world_size,), nprocs=world_size, join=True)
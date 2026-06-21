import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    # Initialize the distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Use NCCL if CUDA is available, otherwise GLOO for CPU
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def worker(rank, world_size):
    setup(rank, world_size)
    
    # Determine device based on availability, adhering to the bug report's context
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16

    # Define shapes for pre-allocation on the receiver side
    shapes = [(5699097, 6, 1), (5699097, 6, 256), (5699097, 256, 1)]

    if rank == 0:
        # Replicate the tensor creation from the bug report
        tensors = [torch.rand(shape, dtype=dtype, device=device) for shape in shapes]

        # Adaptation: Since send_object_list/recv_object_list might not be available in older PyTorch versions,
        # we use the standard dist.send and dist.recv API in a loop.
        for tensor in tensors:
            dist.send(tensor, dst=1)
        print(f"Rank {rank}: Sent objects successfully.")
    else:
        # Pre-allocate tensors to receive data. 
        # Standard dist.recv requires the tensor to be allocated with the correct shape and dtype beforehand.
        tensors = [torch.empty(shape, dtype=dtype, device=device) for shape in shapes]
        
        # Receive the objects on the other rank
        for tensor in tensors:
            dist.recv(tensor, src=0)
        
        # Basic validation to ensure data integrity
        assert tensors[0] is not None
        assert tensors[1] is not None
        assert tensors[2] is not None
        
        assert tensors[0].shape == (5699097, 6, 1)
        assert tensors[1].shape == (5699097, 6, 256)
        assert tensors[2].shape == (5699097, 256, 1)
        
        print(f"Rank {rank}: Received and validated objects successfully.")

    cleanup()

if __name__ == '__main__':
    world_size = 2
    # Spawn 2 processes to simulate distributed environment
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
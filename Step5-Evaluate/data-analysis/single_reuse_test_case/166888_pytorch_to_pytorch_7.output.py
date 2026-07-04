import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    # Initialize the process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Use 'gloo' backend for compatibility, though 'nccl' is preferred for CUDA
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def worker(rank, world_size):
    # Set seed to ensure both processes generate the same random tensors for verification
    torch.manual_seed(0)
    
    setup(rank, world_size)
    
    # Adapted from the original bug report's data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(10, 20, 30, device=device)
    max_val = torch.tensor(5.0, device=device)

    if rank == 0:
        # Check if the newer API exists
        if hasattr(dist, 'send_object_list'):
            dist.send_object_list([x, max_val], dst=1)
        else:
            # Fallback for older PyTorch versions: send tensors individually
            dist.send(x, dst=1)
            dist.send(max_val, dst=1)
            
    elif rank == 1:
        if hasattr(dist, 'recv_object_list'):
            # Receive the objects using the newer API
            req_list = [None, None]
            dist.recv_object_list(req_list, src=0)
            received_x, received_max_val = req_list
        else:
            # Fallback for older PyTorch versions: receive tensors individually
            # We must pre-allocate tensors of the correct shape and type
            received_x = torch.empty_like(x)
            received_max_val = torch.empty_like(max_val)
            
            dist.recv(received_x, src=0)
            dist.recv(received_max_val, src=0)
        
        # Verify the received objects match the original tensors
        assert torch.allclose(received_x, x), "Received x does not match original x"
        assert torch.allclose(received_max_val, max_val), "Received max_val does not match original max_val"
        assert received_max_val.item() == 5.0, "Received max_val value is incorrect"
        
        print(f"Rank {rank}: Test passed. Objects received and verified.")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    # Check for CUDA availability as the original bug was CUDA-specific
    if not torch.cuda.is_available():
        print("CUDA not available. Falling back to CPU for distributed test.")
    
    # Spawn processes to run the worker function
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
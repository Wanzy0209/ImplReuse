import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def f(x, y):
    # The function from the original bug report involving in-place mutation
    x.copy_(x.flip(1))
    y = y.sum(dim=1, keepdim=True) + y
    return x + y

def worker(rank, world_size):
    # Initialize the process group
    dist.init_process_group(
        backend="gloo", # Using gloo for compatibility in minimal test environments
        init_method=f"tcp://127.0.0.1:{29500}",
        rank=rank,
        world_size=world_size
    )
    
    # Set seed to ensure inputs are identical across ranks for verification
    torch.manual_seed(42)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create inputs matching the original bug report's scale
    x = torch.randn(20, 1024 * 1024, device=device)
    y = torch.randn(20, 1024 * 1024, device=device)
    
    if rank == 0:
        # Rank 0 computes the result using the function f
        result = f(x, y)
        obj_list = [result]
    else:
        # Other ranks prepare to receive
        obj_list = [None]
    
    # Adaptation: Replace torch.compile with torch.distributed.broadcast_object_list
    # We test if the API correctly handles the complex tensor data produced by f
    dist.broadcast_object_list(obj_list, src=0)
    
    if rank != 0:
        # Verify that the received object matches the locally computed result
        # This ensures broadcast_object_list preserves numerical correctness
        local_result = f(x, y)
        received_result = obj_list[0]
        
        torch.testing.assert_close(local_result, received_result)
        print(f"Rank {rank}: Test passed. Broadcasted result matches local computation.")
    else:
        print(f"Rank {rank}: Result broadcasted.")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    # Use multiprocessing to simulate a distributed environment
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
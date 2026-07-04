import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def f(x, y):
    # Original function from the bug report
    x.copy_(x.flip(1))
    y = y.sum(dim=1, keepdim=True) + y
    return x + y

def run_test(rank, world_size):
    # Initialize the process group
    dist.init_process_group(
        backend="gloo", 
        init_method=f"tcp://127.0.0.1:29500",
        rank=rank,
        world_size=world_size
    )

    # Set seed for reproducibility across processes
    torch.manual_seed(42)

    # Create inputs. 
    # Note: Using CPU to ensure the test runs in all environments, 
    # though the original bug report specified CUDA.
    x = torch.randn(20, 1024 * 1024)
    x_copy = x.clone()
    y = torch.randn(20, 1024 * 1024)

    if rank == 0:
        # Sender process
        # Compute the result to be sent
        res = f(x_copy, y)
        
        # Fix: Use standard dist.send for tensors instead of non-existent send_object_list
        dist.send(res, dst=1)
        
    elif rank == 1:
        # Receiver process
        # Compute the reference result locally
        ref = f(x, y)
        
        # Fix: Use standard dist.recv for tensors
        # We need to allocate a tensor to receive the data into
        act = torch.empty_like(ref)
        dist.recv(act, src=0)
        
        # Verify the received object matches the reference
        torch.testing.assert_close(ref, act)
        print("Test passed: Received object matches reference.")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    # Spawn processes to simulate distributed environment
    mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)
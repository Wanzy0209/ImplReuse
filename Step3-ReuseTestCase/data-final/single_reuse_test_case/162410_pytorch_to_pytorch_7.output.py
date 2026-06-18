import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def worker(rank, world_size):
    # Initialize the process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    # Use 'gloo' backend for CPU/CUDA compatibility in testing
    # Note: For CUDA tensors, 'nccl' is preferred but requires specific hardware setup.
    # 'gloo' works for both CPU and CUDA tensors in most test environments.
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    
    # Set seed for reproducibility
    torch.manual_seed(42)

    if rank == 0:
        # Sender
        x = torch.randn(20, 1024 * 1024, device=device)
        y = torch.randn(20, 1024 * 1024, device=device)

        # Apply the operations from the original bug report
        # These operations involve in-place modification and views
        x.copy_(x.flip(1))
        y = y.sum(dim=1, keepdim=True) + y

        # Send the objects using the similar API
        # We verify that send_object_list correctly handles these specific tensor states
        dist.send_object_list([x, y], dst=1)

    elif rank == 1:
        # Receiver
        # Generate reference data to compare against
        torch.manual_seed(42)
        x_ref = torch.randn(20, 1024 * 1024, device=device)
        y_ref = torch.randn(20, 1024 * 1024, device=device)

        # Apply the same operations
        x_ref.copy_(x_ref.flip(1))
        y_ref = y_ref.sum(dim=1, keepdim=True) + y_ref

        # Receive objects
        obj_list = [None, None]
        dist.recv_object_list(obj_list, src=0)
        rx, ry = obj_list

        # Verify correctness
        # This ensures that send_object_list preserves data integrity
        # for tensors modified in-place via views (the context of the original bug)
        torch.testing.assert_close(rx, x_ref)
        torch.testing.assert_close(ry, y_ref)
        print("Test Passed: Received tensors match reference.")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    # Check if CUDA is available to set appropriate device count for spawning
    if torch.cuda.is_available():
        # Ensure we have enough GPUs if using NCCL
        assert torch.cuda.device_count() >= world_size, "Not enough GPUs available"
    
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
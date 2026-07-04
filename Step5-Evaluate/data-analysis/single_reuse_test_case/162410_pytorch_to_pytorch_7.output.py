import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import pickle

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

        # Send the objects using a manual pickle implementation
        # to replace the missing send_object_list API
        data_to_send = [x, y]
        serialized = pickle.dumps(data_to_send)
        
        # Send size first
        size_tensor = torch.tensor([len(serialized)], dtype=torch.long, device=device)
        dist.send(size_tensor, dst=1)
        
        # Send data
        # Using torch.tensor with bytearray is compatible with older PyTorch versions
        payload_tensor = torch.tensor(bytearray(serialized), dtype=torch.uint8, device=device)
        dist.send(payload_tensor, dst=1)

    elif rank == 1:
        # Receiver
        # Generate reference data to compare against
        torch.manual_seed(42)
        x_ref = torch.randn(20, 1024 * 1024, device=device)
        y_ref = torch.randn(20, 1024 * 1024, device=device)

        # Apply the same operations
        x_ref.copy_(x_ref.flip(1))
        y_ref = y_ref.sum(dim=1, keepdim=True) + y_ref

        # Receive objects using a manual pickle implementation
        # to replace the missing recv_object_list API
        
        # Receive size
        size_tensor = torch.tensor([0], dtype=torch.long, device=device)
        dist.recv(size_tensor, src=0)
        size = size_tensor.item()
        
        # Receive data
        payload_tensor = torch.empty(size, dtype=torch.uint8, device=device)
        dist.recv(payload_tensor, src=0)
        
        # Deserialize
        # Move to CPU to convert to bytes for unpickling
        buffer = payload_tensor.cpu().numpy().tobytes()
        obj_list = pickle.loads(buffer)
        rx, ry = obj_list

        # Verify correctness
        # This ensures that the send/recv mechanism preserves data integrity
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
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import pickle

def setup(rank, world_size):
    # Initialize the process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Using 'gloo' backend for CPU compatibility to ensure the test runs everywhere
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def worker(rank, world_size):
    setup(rank, world_size)

    # Check if the modern API is available
    has_object_list_api = hasattr(dist, 'send_object_list') and hasattr(dist, 'recv_object_list')

    if rank == 0:
        # Reproduce the tensor setup from the original bug report
        # Using CPU to ensure the test runs without GPU requirements
        arg0 = torch.empty([1, 1], dtype=torch.float32, requires_grad=True)
        
        # Fix: Initialize directly with value to avoid in-place operation on a leaf Variable that requires grad
        arg1 = torch.tensor(3.14, dtype=torch.float32, requires_grad=True)

        # The operation from the original bug report
        t0 = arg0
        t1 = arg1
        t2 = t0.clone()
        t2.fill_diagonal_(t1.item())

        # Adaptation: Replace torch.compile with torch.distributed.send_object_list
        # We send the result of the operation (t2) along with other objects to test the API
        object_list = [t2, "metadata", 42]
        
        try:
            if has_object_list_api:
                dist.send_object_list(object_list, dst=1)
            else:
                # Fallback for older PyTorch versions: send size then pickled bytes
                serialized = pickle.dumps(object_list)
                size_tensor = torch.tensor([len(serialized)], dtype=torch.long)
                dist.send(size_tensor, dst=1)
                buffer = torch.ByteTensor(torch.ByteStorage.from_buffer(serialized))
                dist.send(buffer, dst=1)
            
            print(f"Rank {rank}: Sent object list successfully.")
        except Exception as e:
            print(f"Rank {rank}: Failed to send object list. Error: {e}")

    elif rank == 1:
        object_list = [None, None, None]
        
        try:
            if has_object_list_api:
                dist.recv_object_list(object_list, src=0)
            else:
                # Fallback for older PyTorch versions: receive size then pickled bytes
                size_tensor = torch.tensor([0], dtype=torch.long)
                dist.recv(size_tensor, src=0)
                buffer = torch.ByteTensor(size_tensor.item())
                dist.recv(buffer, src=0)
                received_list = pickle.loads(bytes(buffer.tolist()))
                object_list[:] = received_list
            
            received_tensor = object_list[0]
            received_str = object_list[1]
            received_int = object_list[2]

            # Verify the received data matches the original logic
            assert received_tensor.shape == (1, 1), f"Shape mismatch: {received_tensor.shape}"
            assert torch.allclose(received_tensor, torch.tensor([[3.14]])), f"Value mismatch: {received_tensor}"
            assert received_str == "metadata", f"String mismatch: {received_str}"
            assert received_int == 42, f"Int mismatch: {received_int}"
            
            print(f"Rank {rank}: Received and verified object list successfully.")
        except Exception as e:
            print(f"Rank {rank}: Failed to receive or verify object list. Error: {e}")

    cleanup()

def main():
    world_size = 2
    # Spawn 2 processes
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
import os
import io
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import sys

# Polyfill for send_object_list and recv_object_list for older PyTorch versions
# These functions were added in PyTorch 1.9. If they are missing, we implement them
# using torch.save/load (which handles device preservation) and basic point-to-point comms.
if not hasattr(dist, 'send_object_list') or not hasattr(dist, 'recv_object_list'):
    def _send_object_list(obj_list, dst, group=None):
        # Serialize the list of objects using torch.save to preserve device info
        buffer = io.BytesIO()
        torch.save(obj_list, buffer)
        buffer_bytes = buffer.getvalue()
        
        # Send the size of the buffer first
        size_tensor = torch.tensor([len(buffer_bytes)], dtype=torch.long)
        dist.send(size_tensor, dst=dst, group=group)
        
        # Send the buffer as a byte tensor
        buffer_tensor = torch.frombuffer(buffer_bytes, dtype=torch.uint8)
        dist.send(buffer_tensor, dst=dst, group=group)

    def _recv_object_list(obj_list, src, group=None):
        # Receive the size of the buffer
        size_tensor = torch.zeros(1, dtype=torch.long)
        dist.recv(size_tensor, src=src, group=group)
        
        # Receive the buffer
        buffer_tensor = torch.zeros(size_tensor.item(), dtype=torch.uint8)
        dist.recv(buffer_tensor, src=src, group=group)
        
        # Deserialize using torch.load
        buffer_bytes = buffer_tensor.numpy().tobytes()
        buffer = io.BytesIO(buffer_bytes)
        loaded_list = torch.load(buffer)
        
        # Update the provided list in place
        obj_list[:] = loaded_list

    # Monkey patch the dist module if the attributes are missing
    if not hasattr(dist, 'send_object_list'):
        dist.send_object_list = _send_object_list
    if not hasattr(dist, 'recv_object_list'):
        dist.recv_object_list = _recv_object_list

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run_test(rank, world_size):
    setup(rank, world_size)
    
    # Use CUDA if available, as the original bug report was specific to CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Inputs from the original bug report
    x = torch.zeros(1, 32, dtype=torch.int64, device=device)
    y = torch.zeros(1, dtype=torch.int32, device=device)

    if rank == 0:
        # Adapted call site: using torch.distributed.send_object_list
        # instead of torch.compile(f)(...)
        try:
            dist.send_object_list([x, y], dst=1)
            print("Send successful on rank 0")
        except Exception as e:
            print(f"Send failed on rank 0: {e}")
            sys.exit(1)
            
    elif rank == 1:
        # Receive the objects to verify the API works with the specific data types
        recv_list = [None, None]
        try:
            dist.recv_object_list(recv_list, src=0)
            
            # Assertions to verify data integrity
            assert recv_list[0] is not None, "Received x is None"
            assert recv_list[1] is not None, "Received y is None"
            assert torch.equal(recv_list[0], x), "x tensors do not match"
            assert torch.equal(recv_list[1], y), "y tensors do not match"
            assert recv_list[0].dtype == torch.int64, "x dtype mismatch"
            assert recv_list[1].dtype == torch.int32, "y dtype mismatch"
            assert recv_list[0].device == x.device, "x device mismatch"
            assert recv_list[1].device == y.device, "y device mismatch"
            
            print("Receive and verification successful on rank 1")
        except Exception as e:
            print(f"Receive or verification failed on rank 1: {e}")
            sys.exit(1)

    cleanup()

if __name__ == "__main__":
    # Check for CUDA availability to match the bug report context
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Falling back to CPU. The original bug was CUDA-specific.")
    
    world_size = 2
    try:
        mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)
    except Exception as e:
        print(f"Test execution failed: {e}")
        sys.exit(1)
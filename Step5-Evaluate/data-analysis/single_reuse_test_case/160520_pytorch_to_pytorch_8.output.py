import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import contextlib

def run(rank, world_size):
    # Setup distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    # Note: Using 'gloo' for CPU compatibility in this example, 
    # but the bug context involves CUDA. 
    # If running on a CUDA machine, change backend to 'nccl'.
    backend = 'gloo' 
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    if rank == 0:
        # Sender: Sends a mix of tensor and non-tensor arguments
        # This mimics the "Non-tensor arguments" part of the bug description
        tensor_data = torch.randn(2, 2)
        if torch.cuda.is_available():
            tensor_data = tensor_data.cuda()
            
        objects_to_send = [
            42,             # int (non-tensor)
            3.14,           # float (non-tensor)
            "test_string",  # string (non-tensor)
            tensor_data     # tensor
        ]
        print(f"Rank {rank} sending: {objects_to_send}")
        dist.send_object_list(objects_to_send, dst=1)
        
    elif rank == 1:
        # Receiver: Receives inside a DeviceContext
        # This mimics the "run it under a DeviceContext" part of the bug description
        
        # Initialize empty list to receive data
        recv_list = [None] * 4
        
        # Use Profiler to check for redundant H2D/D2H memcpy
        with torch.profiler.profile(
            with_stack=True, 
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA if torch.cuda.is_available() else torch.profiler.ProfilerActivity.CPU,
            ],
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./log_dist"),
        ) as prof:
            
            # The core of the adaptation: 
            # Running the API inside a torch.device context
            # Fix: torch.device is not a context manager in older PyTorch versions.
            # Use torch.cuda.device for CUDA and nullcontext for CPU.
            device_context = torch.cuda.device(0) if torch.cuda.is_available() else contextlib.nullcontext()
            
            with device_context:
                # Passing device argument explicitly as well
                dist.recv_object_list(
                    recv_list, 
                    src=0, 
                    device="cuda" if torch.cuda.is_available() else None
                )
        
        print(f"Rank {rank} received: {recv_list}")
        
        # Assertions to verify correctness
        # If redundant H2D/D2H occurred, types might change or errors might occur
        assert recv_list[0] == 42, "Integer data mismatch"
        assert recv_list[1] == 3.14, "Float data mismatch"
        assert recv_list[2] == "test_string", "String data mismatch"
        
        # Check tensor device
        if torch.cuda.is_available():
            assert recv_list[3].is_cuda, "Tensor should be on CUDA"
        
        print("Test passed: Data integrity maintained.")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    # Check for CUDA availability to adjust backend
    if torch.cuda.is_available():
        print("CUDA detected. Ensure NCCL is installed for GPU transfer.")
        # Note: For this script to run with NCCL, proper env vars usually needed.
        # We stick to spawn logic.
        
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
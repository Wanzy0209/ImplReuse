import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def test_init_process_group_with_device_id(rank, world_size):
    """
    Test case to verify torch.distributed.init_process_group 
    with the device_id argument, as used in the FSDP2 example.
    """
    # Setup environment variables for the spawned process
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Determine device based on availability
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
    else:
        # Fallback to CPU if CUDA is not available
        device = torch.device("cpu")

    # Logic adapted from the bug report's original call site
    backend = dist.get_default_backend_for_device(device)
    
    # Call the API under test
    dist.init_process_group(backend=backend, device_id=device)

    # Assertions to verify the process group initialized correctly
    assert dist.is_initialized(), "Process group failed to initialize"
    assert dist.get_rank() == rank, f"Rank mismatch: expected {rank}, got {dist.get_rank()}"
    assert dist.get_world_size() == world_size, f"World size mismatch: expected {world_size}, got {dist.get_world_size()}"
    
    # Verify the backend matches the requested device type
    expected_backend = "nccl" if device.type == "cuda" else "gloo"
    # Note: get_backend() returns the enum, so we check the string representation or name
    assert dist.get_backend().lower() == expected_backend, \
        f"Backend mismatch: expected {expected_backend}, got {dist.get_backend()}"

    print(f"Rank {rank}: Initialization successful with backend {dist.get_backend()} on device {device}")

    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    # Use multiprocessing to simulate a distributed environment
    mp.spawn(test_init_process_group_with_device_id, args=(world_size,), nprocs=world_size, join=True)
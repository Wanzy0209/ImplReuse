import torch
import torch.distributed as dist
import multiprocessing as mp
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run(rank, world_size):
    setup(rank, world_size)
    
    # Attempt to use MPS if available, similar to the original bug report context
    # Fallback to CPU to ensure the test is runnable on non-MPS machines
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    torch_device = torch.device(device)

    if rank == 0:
        # Sender process
        # Create tensors on the target device
        tensor1 = torch.randn(2, 2).to(torch_device)
        tensor2 = torch.randn(2, 2).to(torch_device)
        obj_list = [tensor1, tensor2]
        
        # Send objects to rank 1
        dist.send_object_list(obj_list, dst=1)
        
    elif rank == 1:
        # Receiver process
        # Prepare a list to receive objects
        recv_list = [None, None]
        
        # API Under Test: torch.distributed.recv_object_list
        # We pass the device argument to test if the API handles device placement correctly
        dist.recv_object_list(recv_list, src=0, device=torch_device)
        
        # Assertions to verify the test case behavior
        assert recv_list[0] is not None, "First object was not received"
        assert recv_list[1] is not None, "Second object was not received"
        assert recv_list[0].device == torch_device, f"Object 0 not on {device}"
        assert recv_list[1].device == torch_device, f"Object 1 not on {device}"
        
        print(f"Rank {rank} successfully received objects on {device}.")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    # Spawn processes for distributed testing
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
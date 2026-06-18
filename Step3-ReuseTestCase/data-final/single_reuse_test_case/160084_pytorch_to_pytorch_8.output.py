import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def test_recv_object_list(rank, world_size):
    """
    Test case for torch.distributed.recv_object_list.
    Adapted from the original regression test context to verify 
    distributed object reception with CUDA tensors.
    """
    # Setup distributed environment
    # Use nccl if CUDA is available to match the original bug's context, otherwise gloo
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    
    # Initialize the process group
    dist.init_process_group(
        backend,
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{29500}"
    )

    # Set device based on rank if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        # Sender: Create a list of objects similar to the inputs in the original bug
        # Original bug used: torch.randn(4, 10).to(torch_device)
        obj_to_send = [
            torch.randn(4, 10).to(device),
            torch.randn(4, 10).to(device),
            "string_metadata"
        ]
        
        print(f"Rank {rank} sending objects...")
        dist.send_object_list(obj_to_send, dst=1)
        
    elif rank == 1:
        # Receiver: Prepare an empty list to receive data
        recv_list = [None, None, None]
        
        print(f"Rank {rank} receiving objects...")
        # This is the API under test
        dist.recv_object_list(recv_list, src=0)
        
        # Assertions to verify correctness
        assert recv_list[0] is not None, "First tensor not received"
        assert recv_list[1] is not None, "Second tensor not received"
        assert recv_list[2] == "string_metadata", "String metadata mismatch"
        
        assert isinstance(recv_list[0], torch.Tensor), "Received object is not a Tensor"
        assert recv_list[0].shape == (4, 10), f"Shape mismatch: {recv_list[0].shape}"
        assert recv_list[0].device == device, f"Device mismatch: {recv_list[0].device}"
        
        print(f"Rank {rank} successfully received and verified objects.")

    # Clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    # Check for CUDA availability
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Falling back to gloo backend on CPU.")
    
    world_size = 2
    # Spawn processes to simulate distributed environment
    mp.spawn(test_recv_object_list, args=(world_size,), nprocs=world_size)
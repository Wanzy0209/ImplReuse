"""
Run:

torchrun --nproc_per_node=2 test_recv_object_list.py
"""

import torch
import torch.distributed as dist
import os

def test_recv_object_list():
    """
    Tests torch.distributed.recv_object_list in a 2-process setup.
    Rank 0 sends a list of objects (Tensor and metadata), Rank 1 receives them.
    """
    # Initialize the process group
    # Use 'nccl' if CUDA is available, otherwise fallback to 'gloo' for CPU testing
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend)
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    if world_size < 2:
        if rank == 0:
            print("This test requires at least 2 processes. Run with torchrun --nproc_per_node=2")
        return

    # Scenario: Simulating data transfer between pipeline stages
    # Rank 0 acts as the previous stage, Rank 1 as the current stage.
    
    if rank == 0:
        # Create a list of objects to send
        # 1. A Tensor representing activations
        tensor_data = torch.randn(8, 32, device=device)
        # 2. A dictionary representing metadata
        meta_data = {"step": 1, "micro_batch_id": 5}
        
        send_list = [tensor_data, meta_data]
        
        print(f"Rank {rank}: Sending objects...")
        # Send the list of objects to Rank 1
        dist.send_object_list(send_list, dst=1)
        
    elif rank == 1:
        # Prepare a list to receive objects.
        # The list must be pre-allocated with the correct number of elements (None placeholders).
        recv_list = [None, None]
        
        print(f"Rank {rank}: Waiting to receive objects...")
        # Receive the list of objects from Rank 0
        dist.recv_object_list(recv_list, src=0)
        
        # Verify the received data
        received_tensor = recv_list[0]
        received_meta = recv_list[1]
        
        assert isinstance(received_tensor, torch.Tensor), "Expected first element to be a Tensor"
        assert received_tensor.shape == (8, 32), f"Expected Tensor shape (8, 32), got {received_tensor.shape}"
        assert received_tensor.device == device, f"Expected Tensor on device {device}"
        
        assert isinstance(received_meta, dict), "Expected second element to be a dictionary"
        assert received_meta.get("step") == 1, "Metadata step mismatch"
        assert received_meta.get("micro_batch_id") == 5, "Metadata micro_batch_id mismatch"
        
        print(f"Rank {rank}: Successfully received and verified objects.")

    # Synchronize all processes
    dist.barrier()
    
    if rank == 0:
        print("Test Passed.")

if __name__ == "__main__":
    test_recv_object_list()
"""
Run:

python test_send_object_list.py
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def worker(rank, world_size) -> None:
    # Initialize the distributed environment
    # We use TCP initialization to avoid relying on environment variables
    # which are missing when running the script directly without torchrun.
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:29500",
        world_size=world_size,
        rank=rank
    )
    
    # Retrieve rank and world_size to maintain original logic flow
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Ensure we have at least 2 processes for this test
    if world_size < 2:
        if rank == 0:
            print("This test requires at least 2 processes (nproc_per_node=2).")
        return

    # Setup device similar to the original bug report
    device = torch.device("cuda", rank)

    # Adaptation: Replace the pipeline schedule step with torch.distributed.send_object_list
    # Rank 0 acts as the sender
    if rank == 0:
        # Create a list of objects to send (tensors, dicts, strings)
        # This mimics the data types often passed in pipeline parallelism
        tensor_data = torch.randint(0, 128, (8, 4096), device=device)
        metadata = {"step": 1, "stage": "input"}
        string_data = "pipeline_token"

        object_list = [tensor_data, metadata, string_data]

        # Call the API under test
        dist.send_object_list(object_list, dst=1)
        print(f"Rank {rank}: Successfully sent object list.")

    # Rank 1 acts as the receiver
    elif rank == 1:
        # Prepare a list to receive objects. 
        # The list must be pre-allocated with the correct number of elements.
        received_objects = [None, None, None]

        # Receive objects
        dist.recv_object_list(received_objects, src=0)

        # Assertions to verify correctness
        assert isinstance(received_objects[0], torch.Tensor), "Expected first object to be a Tensor"
        assert received_objects[0].shape == (8, 4096), "Tensor shape mismatch"
        assert received_objects[1] == {"step": 1, "stage": "input"}, "Metadata mismatch"
        assert received_objects[2] == "pipeline_token", "String data mismatch"

        print(f"Rank {rank}: Successfully received and verified object list.")

    dist.destroy_process_group()

def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA is not available. This test requires GPUs.")
        return

    # Use torch.multiprocessing.spawn to launch the distributed processes
    # This sets up the environment (rank, world_size) programmatically
    # instead of relying on torchrun.
    world_size = 2
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
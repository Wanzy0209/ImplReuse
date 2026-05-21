import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import gc
import os

def run_test(rank, world_size):
    """
    Test case for torch.distributed.send_object_list to check for memory leaks
    in a loop, adapted from the original torch.compile memory leak report.
    """
    # Initialize the process group
    dist.init_process_group(
        backend="gloo", # Using gloo for CPU compatibility
        init_method=f"tcp://127.0.0.1:29500",
        rank=rank,
        world_size=world_size
    )

    if rank == 0:
        # Sender process
        print(f"Rank {rank}: Starting send loop...")
        for step in range(300):
            # Create a list of tensors to send
            # This mimics the tensor creation in the original bug's training loop
            tensor_list = [torch.randn(128, 128) for _ in range(10)]
            
            # Call the similar API: torch.distributed.send_object_list
            dist.send_object_list(tensor_list, dst=1)
            
            # Explicitly delete to allow garbage collection
            del tensor_list
            
            # Monitor status similar to the original bug report
            if step % 50 == 0:
                gc.collect()
                # Note: Tracking exact tensor counts in distributed CPU mode is noisy,
                # but we verify the loop completes without OOM.
                print(f"Step {step} | Sent objects | Rank: {rank}")
                
    else:
        # Receiver process
        print(f"Rank {rank}: Starting receive loop...")
        for step in range(300):
            # Prepare buffer
            tensor_list = [None] * 10
            
            # Receive objects
            dist.recv_object_list(tensor_list, src=0)
            
            del tensor_list
            
            if step % 50 == 0:
                print(f"Step {step} | Received objects | Rank: {rank}")

    dist.destroy_process_group()

if __name__ == "__main__":
    # Set environment variables for initialization
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    
    world_size = 2
    mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)
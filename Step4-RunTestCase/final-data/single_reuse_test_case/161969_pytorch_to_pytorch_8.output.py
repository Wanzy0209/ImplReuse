import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def example_recv_function():
    """
    Wrapper function for the API under test, adapted to be compiled.
    """
    obj_list = [None]
    # Call the similar API: torch.distributed.recv_object_list
    dist.recv_object_list(obj_list, src=0)
    return obj_list[0]

def run(rank, world_size):
    # Initialize distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    try:
        # Check for API availability to handle older PyTorch versions gracefully
        if not (hasattr(dist, 'send_object_list') and hasattr(dist, 'recv_object_list')):
            if rank == 0:
                print("Skipping test: send_object_list/recv_object_list not available in this PyTorch version.")
            return

        # Use MPS if available, as the original bug was specific to MPS
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            if rank == 0:
                print("MPS not available, falling back to CPU for test.")

        if rank == 0:
            # Sender process
            # Create a tensor similar to the bug report context
            data = torch.randn(2, 5, 3, device=device)
            # Send the object list
            dist.send_object_list([data], dst=1)
        else:
            # Receiver process
            # Adaptation: Compile the function containing the similar API call
            # This mimics the original bug report's usage of torch.compile
            compiled_function = torch.compile(example_recv_function)
            
            try:
                received_obj = compiled_function()
                
                # Verify the received object
                assert isinstance(received_obj, torch.Tensor)
                assert received_obj.shape == (2, 5, 3)
                
                # Check contiguity, inspired by the bug report's assertion failure
                # The original bug failed on is_contiguous() check inside compiled function
                is_contig = received_obj.is_contiguous()
                print(f"Rank {rank}: Test passed. Received tensor is_contiguous: {is_contig}")
                
            except RuntimeError as e:
                # Catch potential errors similar to the bug report
                print(f"Rank {rank}: RuntimeError during compiled execution: {e}")
    finally:
        dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    # Spawn processes to run the distributed test
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
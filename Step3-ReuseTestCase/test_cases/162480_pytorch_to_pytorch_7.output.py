import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    # Initialize the distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Use 'gloo' backend for CPU-based testing
    dist.init_process_group(
        backend='gloo',
        rank=rank,
        world_size=world_size
    )

def cleanup():
    dist.destroy_process_group()

def test_send_object_list_floats(rank, world_size):
    """
    Test that torch.distributed.send_object_list handles float inputs correctly.
    This mirrors the concern from the bug report regarding float handling,
    ensuring the distributed API robustly processes float values.
    """
    setup(rank, world_size)
    
    if rank == 0:
        # Original bug context: rebind_unbacked failed on float inputs.
        # Here we verify send_object_list handles a list containing floats.
        object_list = [1.5, 2.5, 3.14, 0.001]
        
        try:
            dist.send_object_list(object_list, dst=1)
            print(f"Rank {rank}: Successfully sent list containing floats: {object_list}")
        except Exception as e:
            print(f"Rank {rank}: Failed to send object list. Error: {e}")
            
    elif rank == 1:
        # Prepare a buffer to receive the objects
        recv_buffer = [None] * 4
        
        try:
            dist.recv_object_list(recv_buffer, src=0)
            print(f"Rank {rank}: Successfully received list: {recv_buffer}")
            
            # Verify the received floats match the expected values
            expected = [1.5, 2.5, 3.14, 0.001]
            assert recv_buffer == expected, f"Mismatch: expected {expected}, got {recv_buffer}"
            print(f"Rank {rank}: Assertion passed. Floats handled correctly.")
        except Exception as e:
            print(f"Rank {rank}: Failed to receive or verify object list. Error: {e}")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    # Spawn 2 processes to simulate the distributed environment
    mp.spawn(test_send_object_list_floats, args=(world_size,), nprocs=world_size, join=True)
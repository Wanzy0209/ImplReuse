import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def test_recv_object_list(rank, world_size):
    """
    Test case for torch.distributed.recv_object_list.
    
    This test verifies that a list of picklable objects can be received 
    correctly from another rank. It adapts the context of running a PyTorch 
    operation (originally a C++ extension load) to a distributed communication 
    operation.
    """
    # Setup environment for distributed communication
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    # Initialize the process group
    # Using 'gloo' backend as it is generally available for CPU-based object transfer
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    if rank == 0:
        # Rank 0 acts as the sender to facilitate the test
        objects_to_send = [
            "test_string", 
            42, 
            3.14, 
            torch.tensor([1, 2, 3])
        ]
        dist.send_object_list(objects_to_send, dst=1)
        
    elif rank == 1:
        # Rank 1 acts as the receiver
        # This is the API under test: torch.distributed.recv_object_list
        recv_buffer = [None] * 4
        
        # Replacing the original torch.utils.cpp_extension.load call with 
        # torch.distributed.recv_object_list
        dist.recv_object_list(recv_buffer, src=0)

        # Assertions to verify the received data matches the sent data
        assert recv_buffer[0] == "test_string", "String mismatch"
        assert recv_buffer[1] == 42, "Integer mismatch"
        assert recv_buffer[2] == 3.14, "Float mismatch"
        assert torch.equal(recv_buffer[3], torch.tensor([1, 2, 3])), "Tensor mismatch"
        
        print("Test passed: recv_object_list successfully received the expected objects.")

    # Clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    # Spawn processes to simulate a multi-rank environment
    mp.spawn(test_recv_object_list, args=(world_size,), nprocs=world_size, join=True)
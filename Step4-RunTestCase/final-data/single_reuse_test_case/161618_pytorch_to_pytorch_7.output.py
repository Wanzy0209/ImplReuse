import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import pickle

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def recv_object_list(obj_list, src):
    """
    Helper function to receive a list of objects sent via dist.send_object_list.
    This implements the receiving logic that matches the protocol of send_object_list.
    """
    # 1. Receive the size of the serialized data
    # send_object_list sends a LongTensor of size 1 containing the size
    size_tensor = torch.zeros(1, dtype=torch.long)
    dist.recv(size_tensor, src=src)
    size = size_tensor.item()
    
    # 2. Receive the serialized data
    # It is sent as a ByteTensor
    buffer_tensor = torch.zeros(size, dtype=torch.uint8)
    dist.recv(buffer_tensor, src=src)
    
    # 3. Deserialize the data
    # Convert the byte tensor back to a list of objects
    received_list = pickle.loads(buffer_tensor.numpy().tobytes())
    
    # Update the provided list in place
    obj_list[:] = received_list

def run_test(rank, world_size):
    setup(rank, world_size)

    # Adapted tensor shapes from the original bug report
    m = 20120
    k = 1536
    n = 512

    if rank == 0:
        # Sender
        # Create tensors similar to the original test case
        a = torch.randn((m, n))
        mat1 = torch.randn((m, k))
        mat2 = torch.randn((k, n))
        
        # Create a list of objects to send
        object_list = [a, mat1, mat2, "test_string"]
        
        # Send the list to rank 1
        dist.send_object_list(object_list, dst=1)
        print(f"Rank {rank}: Sent object list containing tensors and string.")

    elif rank == 1:
        # Receiver
        # Create a list to receive data. 
        # Note: The receiver must provide a list of equal size to the sender.
        recv_list = [None, None, None, None]
        
        # Receive the list from rank 0 using the helper function
        recv_object_list(recv_list, src=0)
        
        # Verify the received data
        assert isinstance(recv_list[0], torch.Tensor), "First element is not a Tensor"
        assert recv_list[0].shape == (m, n), f"Shape mismatch for tensor a: {recv_list[0].shape} vs {(m, n)}"
        
        assert isinstance(recv_list[1], torch.Tensor), "Second element is not a Tensor"
        assert recv_list[1].shape == (m, k), f"Shape mismatch for tensor mat1: {recv_list[1].shape} vs {(m, k)}"
        
        assert isinstance(recv_list[2], torch.Tensor), "Third element is not a Tensor"
        assert recv_list[2].shape == (k, n), f"Shape mismatch for tensor mat2: {recv_list[2].shape} vs {(k, n)}"
        
        assert recv_list[3] == "test_string", "String mismatch"
        
        print(f"Rank {rank}: Received and verified object list successfully.")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    # Spawn processes to simulate distributed environment
    mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)
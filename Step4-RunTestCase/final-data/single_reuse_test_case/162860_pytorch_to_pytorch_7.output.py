import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import pickle

# Helper functions to implement send/recv object list functionality
# for PyTorch versions where these APIs are missing.
def send_object_list(obj_list, dst):
    # Serialize the list of objects
    buffer = pickle.dumps(obj_list)
    # Convert bytes to a list of integers to create a tensor
    # This ensures compatibility across different PyTorch versions
    int_list = list(buffer)
    tensor = torch.tensor(int_list, dtype=torch.uint8)
    
    # Send the size of the tensor first
    size = torch.tensor([len(tensor)], dtype=torch.long)
    dist.send(size, dst=dst)
    
    # Send the actual tensor data
    dist.send(tensor, dst=dst)

def recv_object_list(obj_list, src):
    # Receive the size of the incoming tensor
    size = torch.tensor([0], dtype=torch.long)
    dist.recv(size, src=src)
    
    # Allocate a tensor of the appropriate size
    tensor = torch.zeros([size.item()], dtype=torch.uint8)
    
    # Receive the tensor data
    dist.recv(tensor, src=src)
    
    # Convert tensor back to bytes and deserialize
    buffer = bytes(tensor.tolist())
    obj_list[:] = pickle.loads(buffer)

def inner(x):
    # Helper function to modify data, similar to the original test case
    return x + 1

def run_test(rank, world_size):
    # Setup for distributed communication
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    if rank == 0:
        # Sender logic
        x = torch.ones(3)
        # Apply the inner function logic
        x = inner(x)
        
        # Prepare the list of objects to send
        object_list = [x, "test_string"]
        
        # Call the helper function to send objects
        send_object_list(object_list, dst=1)
        
    elif rank == 1:
        # Receiver logic
        object_list = [None, None]
        
        # Receive the objects using the helper function
        recv_object_list(object_list, src=0)
        
        # Assertions to verify the data received matches the expected transformation
        assert torch.equal(object_list[0], torch.ones(3) + 1), "Tensor value mismatch"
        assert object_list[1] == "test_string", "String value mismatch"
        print("Test passed successfully on rank 1.")

    # Clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    # Spawn processes to run the distributed test
    mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)
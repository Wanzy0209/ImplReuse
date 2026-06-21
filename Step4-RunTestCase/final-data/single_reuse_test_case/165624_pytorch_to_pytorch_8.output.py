import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import pickle

# Polyfill for torch.distributed.send_object_list and recv_object_list
# for older PyTorch versions where these APIs might not exist.
def _send_object_list(obj_list, dst):
    """
    Sends a list of picklable objects to the destination rank.
    """
    # Serialize the list of objects
    buffer = pickle.dumps(obj_list)
    # Convert bytes to a tensor
    # Using list() conversion for compatibility across PyTorch versions without numpy dependency
    tensor = torch.ByteTensor(list(buffer))
    # Send the size of the tensor first
    size = torch.tensor([tensor.numel()], dtype=torch.long)
    dist.send(size, dst=dst)
    # Send the actual tensor
    dist.send(tensor, dst=dst)

def _recv_object_list(obj_list, src):
    """
    Receives a list of picklable objects from the source rank.
    """
    # Receive the size of the tensor
    size = torch.tensor([0], dtype=torch.long)
    dist.recv(size, src=src)
    size_val = size.item()
    
    # Allocate buffer and receive tensor
    tensor = torch.empty([size_val], dtype=torch.uint8)
    dist.recv(tensor, src=src)
    
    # Convert tensor back to bytes and deserialize
    buffer = bytes(tensor.tolist())
    received_list = pickle.loads(buffer)
    
    # Update the provided list in place
    obj_list[:] = received_list

# Monkey patch if missing
if not hasattr(dist, 'send_object_list'):
    dist.send_object_list = _send_object_list
if not hasattr(dist, 'recv_object_list'):
    dist.recv_object_list = _recv_object_list

def test_recv_object_list(rank, world_size):
    """
    Test case for torch.distributed.recv_object_list.
    Adapted from the context of verifying API functionality.
    """
    # Initialize process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Use gloo backend for CPU-based testing
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    if rank == 0:
        # Sender process
        # Create a list of picklable objects
        tensor_to_send = torch.tensor([1, 2, 3])
        object_list = [tensor_to_send, "hello_world", {"key": 42}]
        
        # Send the list to rank 1
        dist.send_object_list(object_list, dst=1)
        
    elif rank == 1:
        # Receiver process
        # Initialize a list to receive data
        recv_list = [None, None, None]
        
        # Adapted call site: Replacing the original GraphTransformObserver call
        # with the similar API torch.distributed.recv_object_list
        dist.recv_object_list(recv_list, src=0)
        
        # Assertions to verify the received data matches the sent data
        assert isinstance(recv_list[0], torch.Tensor), "Expected Tensor at index 0"
        assert torch.equal(recv_list[0], torch.tensor([1, 2, 3])), "Tensor values mismatch"
        
        assert recv_list[1] == "hello_world", "String value mismatch"
        
        assert isinstance(recv_list[2], dict), "Expected dict at index 2"
        assert recv_list[2] == {"key": 42}, "Dict value mismatch"

    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(test_recv_object_list, args=(world_size,), nprocs=world_size, join=True)
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import pickle

# Polyfill for send_object_list and recv_object_list for older PyTorch versions
# where these attributes do not exist.
if not hasattr(dist, 'send_object_list'):
    def _send_object_list(obj_list, dst, group=None):
        # Serialize the list of objects using pickle
        buffer = pickle.dumps(obj_list)
        # Create a byte tensor to hold the serialized data
        tensor = torch.ByteTensor(buffer)
        # Send the size of the tensor first so the receiver knows how much to allocate
        size_tensor = torch.tensor([tensor.numel()], dtype=torch.long)
        dist.send(size_tensor, dst=dst, group=group)
        # Send the actual data tensor
        dist.send(tensor, dst=dst, group=group)

    def _recv_object_list(obj_list, src, group=None):
        # Receive the size of the incoming tensor
        size_tensor = torch.tensor([0], dtype=torch.long)
        dist.recv(size_tensor, src=src, group=group)
        size = size_tensor.item()
        
        # Allocate a tensor to receive the data
        tensor = torch.ByteTensor(size)
        # Receive the data
        dist.recv(tensor, src=src, group=group)
        
        # Deserialize the data back into a list of objects
        received_list = pickle.loads(tensor.numpy().tobytes())
        
        # Update the provided list in place to match the behavior of the official API
        obj_list[:] = received_list

    # Monkey patch the functions into torch.distributed
    dist.send_object_list = _send_object_list
    dist.recv_object_list = _recv_object_list

def setup(rank, world_size):
    # Initialize the distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run_test(rank, world_size):
    setup(rank, world_size)

    if rank == 0:
        # Sender process
        # Create a list of objects to send
        objects_to_send = [
            torch.tensor([1, 2, 3], dtype=torch.float32),
            "hello world",
            {"key": "value"}
        ]
        print(f"Rank {rank} sending objects...")
        dist.send_object_list(objects_to_send, dst=1)
    elif rank == 1:
        # Receiver process
        # Prepare a list to receive objects (must match the size of the sent list)
        received_objects = [None] * 3
        
        print(f"Rank {rank} receiving objects...")
        # Call the similar API: torch.distributed.recv_object_list
        dist.recv_object_list(received_objects, src=0)
        
        # Verify the received data
        assert torch.equal(received_objects[0], torch.tensor([1, 2, 3], dtype=torch.float32))
        assert received_objects[1] == "hello world"
        assert received_objects[2] == {"key": "value"}
        print(f"Rank {rank} successfully received and verified objects.")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    # Spawn 2 processes to simulate the distributed environment
    mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)
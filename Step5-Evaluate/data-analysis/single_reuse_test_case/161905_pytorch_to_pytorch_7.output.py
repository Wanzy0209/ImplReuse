import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import pickle

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initialize the process group using the gloo backend for CPU compatibility
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# Compatibility layer for older PyTorch versions that might not have send_object_list/recv_object_list
if not hasattr(dist, 'send_object_list'):
    def _send_object_list_compat(object_list, dst):
        # Serialize the object list
        serialized = pickle.dumps(object_list)
        length = len(serialized)
        
        # Send the length of the serialized data first
        length_tensor = torch.tensor([length], dtype=torch.long)
        dist.send(length_tensor, dst=dst)
        
        # Send the serialized data as a byte tensor
        # Using torch.ByteTensor for compatibility with older PyTorch versions
        data_tensor = torch.ByteTensor(list(serialized))
        dist.send(data_tensor, dst=dst)
    
    dist.send_object_list = _send_object_list_compat

if not hasattr(dist, 'recv_object_list'):
    def _recv_object_list_compat(object_list, src):
        # Receive the length of the serialized data first
        length_tensor = torch.tensor([0], dtype=torch.long)
        dist.recv(length_tensor, src=src)
        length = length_tensor.item()
        
        # Receive the serialized data
        data_tensor = torch.ByteTensor(length)
        dist.recv(data_tensor, src=src)
        
        # Deserialize the data
        received_list = pickle.loads(bytes(data_tensor.tolist()))
        
        # Update the object_list in place
        object_list[:] = received_list

    dist.recv_object_list = _recv_object_list_compat

def run_distributed_test(rank, world_size):
    setup(rank, world_size)

    # Adapted call site logic replacing the original training loop
    if rank == 0:
        # Create data similar to the original bug report context
        images = torch.randn(4, 3, 224, 224)
        labels = torch.randint(0, 10, (4,))
        object_list = [images, labels]
        
        # Call the API under test: torch.distributed.send_object_list
        dist.send_object_list(object_list, dst=1)
        print(f"Rank {rank}: Successfully sent object list.")
        
    elif rank == 1:
        # Prepare list to receive objects
        object_list = [None, None]
        
        # Receive the objects sent by rank 0
        dist.recv_object_list(object_list, src=0)
        print(f"Rank {rank}: Successfully received object list.")
        
        # Assertions to verify the data integrity
        assert object_list[0] is not None, "Received image tensor is None"
        assert object_list[1] is not None, "Received label tensor is None"
        assert object_list[0].shape == (4, 3, 224, 224), f"Shape mismatch for images: {object_list[0].shape}"
        assert object_list[1].shape == (4,), f"Shape mismatch for labels: {object_list[1].shape}"

    cleanup()

def main():
    world_size = 2
    # Spawn processes to simulate a distributed environment
    mp.spawn(run_distributed_test, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
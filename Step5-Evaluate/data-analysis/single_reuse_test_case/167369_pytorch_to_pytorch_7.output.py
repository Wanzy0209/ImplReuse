import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import pickle

# Adapted from the original bug report's user-defined object
class Config:
    def __repr__(self):
        return "Config()"

    def __eq__(self, other):
        # Helper for assertion
        return isinstance(other, Config)

def worker(rank, world_size):
    # Initialize the process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    if rank == 0:
        # Sender
        config = Config()
        
        # Manual serialization to replace send_object_list
        # which is missing in this PyTorch version
        data = pickle.dumps(config)
        size_tensor = torch.tensor([len(data)], dtype=torch.long)
        data_tensor = torch.ByteTensor(data)
        
        dist.send(size_tensor, dst=1)
        dist.send(data_tensor, dst=1)
        
        print(f"Rank {rank} sent object: {config}")
    else:
        # Receiver
        # Manual deserialization to replace recv_object_list
        size_tensor = torch.tensor([0], dtype=torch.long)
        dist.recv(size_tensor, src=0)
        size = size_tensor.item()
        
        data_tensor = torch.ByteTensor(size)
        dist.recv(data_tensor, src=0)
        
        received_config = pickle.loads(data_tensor.numpy().tobytes())
        
        # Verify the object was received correctly
        assert isinstance(received_config, Config), "Received object is not of type Config"
        print(f"Rank {rank} received object: {received_config}")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    # Use multiprocessing to simulate a distributed environment
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
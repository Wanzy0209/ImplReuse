import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

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
        object_list = [config]
        
        # Adapted call site: Using torch.distributed.send_object_list
        # instead of torch.compile with repr()
        dist.send_object_list(object_list, dst=1)
        print(f"Rank {rank} sent object: {object_list[0]}")
    else:
        # Receiver
        object_list = [None]
        dist.recv_object_list(object_list, src=0)
        
        # Verify the object was received correctly
        assert isinstance(object_list[0], Config), "Received object is not of type Config"
        print(f"Rank {rank} received object: {object_list[0]}")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    # Use multiprocessing to simulate a distributed environment
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
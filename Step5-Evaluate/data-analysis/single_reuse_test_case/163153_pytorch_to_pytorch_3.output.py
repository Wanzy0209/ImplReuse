import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

# Simplified ModelArgs from the bug report context
class ModelArgs:
    def __init__(self, n_layers, n_heads, dim):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = dim

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

def run_test(rank, world_size):
    # Adapted from original: init_process_group
    # Using 'gloo' backend for CPU compatibility in this test case
    backend = 'gloo'
    
    # Fix: Explicitly set init_method to 'tcp://' to avoid the missing MASTER_ADDR error
    # which occurs when the default 'env://' method is used without environment variables.
    dist.init_process_group(
        backend=backend, 
        init_method=f'tcp://127.0.0.1:29500',
        rank=rank, 
        world_size=world_size
    )

    # Test Case for torch.distributed.broadcast_object_list
    # Context: In distributed training (like FSDP), it is common to broadcast
    # configuration objects (like ModelArgs) from rank 0 to all other ranks
    # to ensure consistency before model initialization.

    object_list = [None]

    if rank == 0:
        # Rank 0 creates the configuration object
        model_args = ModelArgs(n_layers=10, n_heads=8, dim=4096)
        object_list = [model_args]
        print(f"Rank {rank} broadcasting object: {model_args.__dict__}")
    else:
        # Other ranks initialize with None
        object_list = [None]

    # Call the similar API: torch.distributed.broadcast_object_list
    dist.broadcast_object_list(object_list, src=0)

    # Verification
    assert object_list[0] is not None, f"Rank {rank} failed to receive object"
    assert isinstance(object_list[0], ModelArgs), f"Rank {rank} received wrong object type"
    assert object_list[0] == ModelArgs(n_layers=10, n_heads=8, dim=4096), \
        f"Rank {rank} received incorrect data: {object_list[0].__dict__}"

    print(f"Rank {rank} successfully verified broadcast_object_list.")

    dist.destroy_process_group()

def main():
    world_size = 2
    mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
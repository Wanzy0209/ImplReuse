import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

@torch.compile()
def test(object_list, src):
    # Adapted from original test: calling the target API inside torch.compile
    # Note: recv_object_list does not support autograd, so gradient checks 
    # from the original bug report are not applicable here.
    dist.recv_object_list(object_list, src=src)
    return object_list

def run(rank, world_size):
    setup(rank, world_size)

    if rank == 0:
        # Sender process
        # Create a list of tensors to send
        data = [torch.randn(2, 2) for _ in range(3)]
        dist.send_object_list(data, dst=1)
    else:
        # Receiver process
        # Prepare a list to receive into
        object_list = [None] * 3
        
        # Call the compiled function
        # This mimics the structure of the original test where the API was called under torch.compile
        received = test(object_list, src=0)
        
        # Verify the result
        assert len(received) == 3
        assert all(isinstance(t, torch.Tensor) for t in received)
        print(f"Rank {rank}: Successfully received {len(received)} objects inside torch.compile")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    # Use multiprocessing to simulate a distributed environment
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
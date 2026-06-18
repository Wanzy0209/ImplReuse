import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Use gloo backend for CPU compatibility
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run_test(rank, world_size):
    setup(rank, world_size)
    
    # Adapted data setup from the original bug report
    d = 65
    # Using CPU tensors to ensure the test runs in generic environments
    x = torch.randn((1, 2, 32, 32))

    if rank == 0:
        # Sender
        # Adapted loop from original bug report
        for _ in range(10):
            dist.send_object_list([x], dst=1)
    elif rank == 1:
        # Receiver
        for _ in range(10):
            recv_list = [None]
            # The call site for the similar API
            dist.recv_object_list(recv_list, src=0)
            
            # Assertions to verify the API works correctly
            assert recv_list[0] is not None
            assert isinstance(recv_list[0], torch.Tensor)
            assert recv_list[0].shape == x.shape

    cleanup()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)
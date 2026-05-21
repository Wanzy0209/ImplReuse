import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run(rank, world_size):
    setup(rank, world_size)

    if rank == 0:
        # Sender process
        # Create the int64 tensor similar to the one in the bug report (iota/arange)
        # The bug involved int64 + arange + max, we test the data transfer of int64
        tensor_to_send = torch.arange(36, dtype=torch.int64)
        dist.send_object_list([tensor_to_send], dst=1)
    else:
        # Receiver process
        # Adapt the original call site: use torch.compile on the function containing the similar API
        @torch.compile
        def f_recv():
            obj_list = [None]
            # Call the similar API: torch.distributed.recv_object_list
            dist.recv_object_list(obj_list, src=0)
            return obj_list[0]

        received = f_recv()
        
        # Verify the received tensor matches the expected int64 tensor
        expected = torch.arange(36, dtype=torch.int64)
        assert torch.equal(received, expected), f"Mismatch: received {received}, expected {expected}"

    cleanup()

if __name__ == "__main__":
    world_size = 2
    # Use spawn to launch processes for distributed testing
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
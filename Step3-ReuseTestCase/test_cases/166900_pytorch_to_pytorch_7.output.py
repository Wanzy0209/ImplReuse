import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

# Class definition from the bug report
class Bar:
    def __eq__(self, other):
        return super().__eq__(other)

    def __hash__(self):
        return 0

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
        # Adapted call site: Use send_object_list with the object from the bug report
        # instead of the original torch.compile context
        obj = Bar()
        dist.send_object_list([obj], dst=1)
    else:
        # Receive the object to verify the similar API handles it correctly
        recv = [None]
        dist.recv_object_list(recv, src=0)
        assert isinstance(recv[0], Bar), "Received object is not of type Bar"

    cleanup()

if __name__ == "__main__":
    world_size = 2
    # Spawn two processes to run the distributed test
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
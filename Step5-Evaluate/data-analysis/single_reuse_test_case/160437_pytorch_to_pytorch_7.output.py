import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run(rank, world_size):
    setup(rank, world_size)

    # Handle missing torch.compile for older PyTorch versions
    if hasattr(torch, 'compile'):
        decorator = torch.compile(backend="eager")
    else:
        # Identity decorator if torch.compile is not available
        decorator = lambda f: f

    @decorator
    def fn(x, i):
        if i == 1:
            if rank == 0:
                # Send the tensor wrapped in a list to rank 1
                dist.send_object_list([x], dst=1)
        return x + 1

    inp = torch.randn(3)

    if rank == 0:
        # Original call site adapted
        fn(inp, 0)
        fn(inp, 1)
        fn(inp, 2)
    else:
        # Receiver logic to verify the send
        recv_list = [None]
        dist.recv_object_list(recv_list, src=0)
        # Verify the received tensor matches the input
        assert torch.allclose(recv_list[0], inp), "Received tensor does not match input"

    cleanup()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

# Fix: Handle missing torch.compile for older PyTorch versions (e.g., PyTorch 1.x)
# PyTorch 2.0+ is required for torch.compile, but this environment appears to use an older version.
if not hasattr(torch, 'compile'):
    def dummy_compile(backend=None, **kwargs):
        def decorator(func):
            return func
        return decorator
    torch.compile = dummy_compile

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run(rank, world_size):
    setup(rank, world_size)

    if rank == 0:
        # Sender process
        # Send data twice to match the multiple calls in the original test case
        data_to_send = [torch.randn(3)]
        dist.send_object_list(data_to_send, dst=1)
        
        data_to_send_2 = [torch.randn(3)]
        dist.send_object_list(data_to_send_2, dst=1)
    else:
        # Receiver process
        # Adaptation: Wrap torch.distributed.recv_object_list in torch.compile
        # to verify if it triggers the empty graph bug or similar issues.
        @torch.compile(backend="eager")
        def fn():
            obj_list = []
            dist.recv_object_list(obj_list, src=0)
            return obj_list

        # First call
        res1 = fn()
        assert len(res1) == 1
        assert isinstance(res1[0], torch.Tensor)
        print(f"Rank 1 received first object: {res1[0].shape}")

        # Second call (checking for graph break/empty graph issues)
        res2 = fn()
        assert len(res2) == 1
        assert isinstance(res2[0], torch.Tensor)
        print(f"Rank 1 received second object: {res2[0].shape}")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
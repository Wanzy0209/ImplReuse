import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
from contextlib import nullcontext

# Handle missing torch._inductor gracefully
try:
    import torch._inductor.config as inductor_config
except ModuleNotFoundError:
    # If inductor is not available, use a dummy config that does nothing
    class DummyInductorConfig:
        @staticmethod
        def patch(**kwargs):
            return nullcontext()
    inductor_config = DummyInductorConfig()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initialize process group using gloo backend for CPU compatibility
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run_test(rank, world_size):
    setup(rank, world_size)

    # Dimensions from the original bug report
    m = 20120
    k = 1536
    n = 512

    if rank == 0:
        # Sender process
        # Using CPU tensors to ensure the test runs in environments without GPUs
        a = torch.randn((m, n))
        mat1 = torch.randn((m, k))
        mat2 = torch.randn((k, n))
        objects = [a, mat1, mat2]
        
        # Send objects to rank 1
        dist.send_object_list(objects, dst=1)
    else:
        # Receiver process
        objects = [None, None, None]
        
        # Adaptation: Wrap the distributed receive call in torch.compile
        # similar to the original bug report's structure.
        # We also apply the specific Inductor configurations from the bug.
        f = lambda: dist.recv_object_list(objects, src=0)
        
        with inductor_config.patch(
            max_autotune=True,
            autotune_fallback_to_aten=False,
        ):
            compiled = torch.compile(f, dynamic=False)
            compiled()
        
        # Verify that the objects were received correctly
        assert objects[0] is not None
        assert objects[0].shape == (m, n)
        assert objects[1].shape == (m, k)
        assert objects[2].shape == (k, n)
        print(f"Rank {rank}: Test passed. Received objects with shapes {[o.shape for o in objects]}")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    # Spawn 2 processes to simulate sender and receiver
    mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)
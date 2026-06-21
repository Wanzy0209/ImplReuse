import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    # Initialize the process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Using 'gloo' backend for CPU compatibility to ensure the test is runnable without GPUs.
    # To verify the specific CUDA GIL behavior mentioned in the bug report, 
    # change backend to 'nccl' and ensure tensors are moved to 'cuda'.
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def dist_reduce_op(x: torch.Tensor, dst: int = 0):
    """Wrapper for torch.distributed.reduce to mimic the function style of the original test case."""
    dist.reduce(x, dst=dst)

def run_test(rank, world_size):
    setup(rank, world_size)

    # Adapted from original main()
    # Original: x = torch.randn(4096, 4096, device='cuda')
    # Using CPU here for generic runnability with 'gloo' backend.
    x = torch.randn(4096, 4096)
    
    # Original loop structure
    for _ in range(10):
        # Original calls: torch_add(x, y), torch_compile_add(x, y), triton_add(x, y)
        # Adapted call: torch.distributed.reduce
        dist_reduce_op(x, dst=0)

    cleanup()

def main():
    world_size = 2
    mp.spawn(run_test,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    main()
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

# Handle missing torch._inductor gracefully
try:
    import torch._inductor.config as inductor_config
except (ImportError, ModuleNotFoundError):
    inductor_config = None

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def test_reduce(rank, world_size):
    setup(rank, world_size)

    # Adapted from the original bug report
    m = 20120
    k = 1536
    n = 512

    a = torch.randn((m, n)).cuda()
    mat1 = torch.randn((m, k)).cuda()
    mat2 = torch.randn((k, n)).cuda()

    # Define the function to compile, incorporating the similar API
    def func(a, mat1, mat2):
        # Original operation
        res = torch.addmm(a, mat1, mat2)
        # Similar API: torch.distributed.reduce
        # Reduce the result to rank 0
        dist.reduce(res, dst=0)
        return res

    # Apply configurations from the original bug
    with inductor_config.patch(
        max_autotune=True,
        max_autotune_gemm_backends="TRITON",
        autotune_fallback_to_aten=False,
    ):
        compiled = torch.compile(func)
        result = compiled(a, mat1, mat2)

    if rank == 0:
        print("Test passed successfully.")

    cleanup()

if __name__ == "__main__":
    # Check for torch._inductor availability
    if inductor_config is None:
        print("Skipping test: torch._inductor module not found.")
    # Check for CUDA availability to run the test
    elif torch.cuda.is_available():
        # This test requires at least 2 GPUs to demonstrate distributed reduce
        world_size = min(torch.cuda.device_count(), 2)
        if world_size >= 2:
            mp.spawn(test_reduce, args=(world_size,), nprocs=world_size, join=True)
        else:
            print(f"Skipping test: requires at least 2 GPUs, found {torch.cuda.device_count()}.")
    else:
        print("Skipping test: CUDA not available.")
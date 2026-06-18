import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    # Initialize the distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run_test(rank, world_size):
    setup(rank, world_size)

    # Adapt the original test case to torch.distributed.all_reduce
    # The original bug involved INT64_MIN and -1 causing an arithmetic exception.
    # We simulate this by having one rank hold INT64_MIN and another hold -1,
    # then performing a PRODUCT reduction (multiplication), which triggers the overflow.
    
    if rank == 0:
        # Rank 0 holds the dividend equivalent (INT64_MIN)
        tensor = torch.full((2, 3), torch.iinfo(torch.int64).min, dtype=torch.int64)
    else:
        # Rank 1 holds the divisor equivalent (-1)
        tensor = torch.full((2, 3), -1, dtype=torch.int64)

    print(f"Rank {rank} input tensor:\n{tensor}")

    # Perform all_reduce with PRODUCT to trigger INT64_MIN * -1
    # This checks if the distributed API handles the overflow similarly or crashes
    try:
        dist.all_reduce(tensor, op=dist.ReduceOp.PRODUCT)
        print(f"Rank {rank} result after all_reduce:\n{tensor}")
    except Exception as e:
        print(f"Rank {rank} encountered exception: {e}")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    # Use multiprocessing to simulate a distributed environment
    mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)
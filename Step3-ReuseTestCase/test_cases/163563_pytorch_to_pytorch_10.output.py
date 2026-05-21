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

def worker(rank, world_size):
    setup(rank, world_size)

    # Adapted from the original test case logic
    # Using CPU to ensure the distributed test runs without requiring multi-GPU setup for this specific API test
    # The shapes and dtypes are preserved to test memory handling with large objects
    arg0 = torch.rand([5699097, 6, 1], dtype=torch.bfloat16, device='cpu')
    arg1 = torch.rand([5699097, 6, 256], dtype=torch.bfloat16, device='cpu')
    arg2 = torch.rand([5699097, 256, 1], dtype=torch.bfloat16, device='cpu')

    def foo(arg0, arg1, arg2):
        t0 = arg0
        t1 = torch.sigmoid(t0)
        t2 = arg1
        t3 = torch.sigmoid(t2)
        t4 = arg2
        t5 = torch.exp(t4)
        t6 = torch.baddbmm(t1, t3, t5)
        t7 = t6.reshape((193, 386, 459))
        return t7

    # Run the eager computation
    output = foo(arg0, arg1, arg2)
    
    # Adaptation: Replace torch.compile call with torch.distributed.gather_object
    # We gather the output tensor objects from all ranks to rank 0
    gather_list = [None] * world_size if rank == 0 else None
    
    try:
        dist.gather_object(output, gather_list, dst=0)
        
        if rank == 0:
            print(f"Rank 0 gathered {len(gather_list)} objects successfully.")
            # Verify the gathered objects
            for i, obj in enumerate(gather_list):
                assert obj.shape == output.shape, f"Shape mismatch for rank {i}"
                assert obj.dtype == output.dtype, f"Dtype mismatch for rank {i}"
            print("Gather Object Test Success! ")
    except Exception as e:
        print(f"Rank {rank} failed during gather_object: {e}")
        raise

    cleanup()

if __name__ == '__main__':
    world_size = 2
    # Start multiprocessing
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
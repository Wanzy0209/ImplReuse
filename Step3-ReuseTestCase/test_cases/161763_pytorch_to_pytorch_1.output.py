import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def worker(rank, world_size):
    # Initialize the process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Adapt the bug scenario to torch.distributed.reduce
    # The original bug involved incorrect handling of uint8 negation (treating 7 as -7 instead of 249)
    # We test if the reduction operation handles the negated uint8 tensor correctly.
    
    c = torch.tensor(7, dtype=torch.uint8)
    # neg(c) on uint8 should wrap around: 256 - 7 = 249
    neg_c = torch.neg(c) 
    
    # We perform a reduction (SUM) of this negated tensor across ranks.
    # Rank 0 and Rank 1 both have neg_c = 249.
    # Expected result at Rank 0: 249 + 249 = 498.
    # If a similar sign bug exists in the reduction backend, it might calculate -7 + -7 = -14.
    
    # Note: dist.reduce modifies the tensor in place at the destination
    dist.reduce(neg_c, dst=0, op=dist.ReduceOp.SUM)

    if rank == 0:
        print(f"Result of reduction: {neg_c.item()}")
        # Assert that the result is the sum of the unsigned values (498)
        # and not the sum of the signed interpretation (-14).
        assert neg_c.item() == 498, (
            f"Expected 498 (249+249), but got {neg_c.item()}. "
            "This might indicate a sign extension bug with uint8 tensors."
        )
        print("Test passed.")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
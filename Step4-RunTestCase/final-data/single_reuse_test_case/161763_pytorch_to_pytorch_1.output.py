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
    
    # We use 200 instead of 7 to avoid overflow collision in the assertion.
    # With 7: Correct sum (498) wraps to 242. Buggy sum (-14) wraps to 242. (Collision)
    # With 200: Correct sum (112) stays 112. Buggy sum (-112) wraps to 144. (Distinct)
    c = torch.tensor(200, dtype=torch.uint8)
    
    # neg(c) on uint8 should wrap around: 256 - 200 = 56
    neg_c = torch.neg(c) 
    
    # We perform a reduction (SUM) of this negated tensor across ranks.
    # Rank 0 and Rank 1 both have neg_c = 56.
    # Expected result at Rank 0: 56 + 56 = 112.
    # If a similar sign bug exists in the reduction backend, it might calculate -56 + -56 = -112.
    # -112 stored in uint8 wraps to 144.
    
    # Note: dist.reduce modifies the tensor in place at the destination
    dist.reduce(neg_c, dst=0, op=dist.ReduceOp.SUM)

    if rank == 0:
        print(f"Result of reduction: {neg_c.item()}")
        # Assert that the result is the sum of the unsigned values (112)
        # and not the sum of the signed interpretation (which would wrap to 144).
        assert neg_c.item() == 112, (
            f"Expected 112 (56+56), but got {neg_c.item()}. "
            "This might indicate a sign extension bug with uint8 tensors."
        )
        print("Test passed.")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
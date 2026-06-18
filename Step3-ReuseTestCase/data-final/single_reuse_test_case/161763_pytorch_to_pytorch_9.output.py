import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run_test(rank, world_size):
    setup(rank, world_size)

    # Replicate the data setup from the original bug report
    torch.manual_seed(0)
    x = torch.randn(2, 2, dtype=torch.float32)
    c = torch.tensor(7, dtype=torch.uint8)

    # The original bug involved incorrect handling of uint8 negation and addition.
    # We test if torch.distributed.broadcast_object_list correctly handles
    # the transmission of these specific tensor types and computations.
    
    if rank == 0:
        # Perform the computation on the source rank
        # torch.neg on uint8 wraps around (7 -> 249)
        neg_c = torch.neg(c) 
        res_add = c + x
        res_neg_add = neg_c + x
        
        # Pack results into a list for broadcasting
        object_list = [neg_c, res_add, res_neg_add]
        print(f"Rank {rank} broadcasting: neg_c={neg_c.item()}, res_add shape={res_add.shape}")
    else:
        object_list = [None, None, None]

    # Broadcast the object list from rank 0 to all other ranks
    dist.broadcast_object_list(object_list, src=0)

    # Verify the received objects on non-source ranks
    if rank != 0:
        received_neg_c, received_res_add, received_res_neg_add = object_list
        
        # Expected values based on eager execution logic
        expected_neg_c = torch.tensor(249, dtype=torch.uint8)
        expected_res_add = c + x
        expected_res_neg_add = expected_neg_c + x

        # Assert that the broadcasted data matches the expected computation
        assert torch.equal(received_neg_c, expected_neg_c), \
            f"Rank {rank}: Mismatch in neg_c. Expected {expected_neg_c}, got {received_neg_c}"
        
        assert torch.allclose(received_res_add, expected_res_add), \
            f"Rank {rank}: Mismatch in res_add"
            
        assert torch.allclose(received_res_neg_add, expected_res_neg_add), \
            f"Rank {rank}: Mismatch in res_neg_add"
            
        print(f"Rank {rank}: All assertions passed.")

    cleanup()

def main():
    world_size = 2
    # Start 2 processes
    mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
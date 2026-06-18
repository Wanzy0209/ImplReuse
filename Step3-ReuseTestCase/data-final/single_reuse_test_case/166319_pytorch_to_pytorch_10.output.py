import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def worker(rank, world_size):
    # Initialize the distributed process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Recreate the specific tensor characteristics from the original bug report
    # Original: var_node_7 = arg_1 # size=(20, 15), stride=(15, 1), dtype=int64
    # We use CPU tensors here for broader compatibility in the distributed test
    arg_1 = torch.as_strided(torch.randint(5, 30, (300,)).to(torch.int64), (20, 15), (15, 1))

    def func_to_test(input_tensor):
        # Adaptation: Replace torch.gather with torch.distributed.gather_object
        # The original code used torch.gather for indexing. Here we test the distributed
        # collective operation with the same tensor input characteristics.
        
        if dist.get_rank() == 0:
            gather_list = [None] * world_size
        else:
            gather_list = None
        
        dist.gather_object(input_tensor, gather_list, dst=0)
        
        return gather_list

    # Test Eager
    print(f"[Rank {rank}] Running Eager...")
    result_eager = func_to_test(arg_1)

    # Test Compiled
    # We use fullgraph=True and dynamic=True to mimic the conditions of the original bug report
    print(f"[Rank {rank}] Running Compiled...")
    compiled_func = torch.compile(func_to_test, fullgraph=True, dynamic=True)
    result_compiled = compiled_func(arg_1)

    # Verification
    if rank == 0:
        print("[Rank 0] Verifying results...")
        assert result_eager is not None
        assert result_compiled is not None
        assert len(result_eager) == world_size
        assert len(result_compiled) == world_size
        
        for i in range(world_size):
            # Check if gathered tensors are equal
            assert torch.equal(result_eager[i], result_compiled[i]), \
                f"Mismatch at index {i} between eager and compiled results"
        print(" Test Passed: Eager and Compiled results match.")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
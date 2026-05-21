import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    """
    Initialize the distributed process group.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Use 'gloo' backend as it is generally available for CPU testing
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    """
    Destroy the distributed process group.
    """
    dist.destroy_process_group()

def worker(rank, world_size):
    setup(rank, world_size)

    # Define a function that uses the similar API: torch.distributed.broadcast_object_list
    def broadcast_logic(rank):
        # Create a list of objects. Only rank 0 has the meaningful data initially.
        obj_list = [f"data_rank_{rank}"] if rank == 0 else [None]
        
        # Broadcast the object list from rank 0 to all other ranks
        dist.broadcast_object_list(obj_list, src=0)
        
        return obj_list

    # Compile the function using the original API: torch.compile
    # This tests the interaction between torch.compile and the similar API.
    # The bug report highlights inconsistencies in cache entries (tlparse) between
    # cache hits and misses. We verify functional consistency here.
    compiled_broadcast_logic = torch.compile(broadcast_logic)

    # --- Run 1: Cache Miss ---
    # The first call triggers compilation (cache miss).
    result_miss = compiled_broadcast_logic(rank)
    expected_data = "data_rank_0"
    
    # Verify correctness on cache miss
    assert result_miss[0] == expected_data, \
        f"Rank {rank} (Cache Miss): Expected '{expected_data}', got '{result_miss[0]}'"

    # --- Run 2: Cache Hit ---
    # The second call should use the compiled cache.
    result_hit = compiled_broadcast_logic(rank)
    
    # Verify correctness on cache hit
    assert result_hit[0] == expected_data, \
        f"Rank {rank} (Cache Hit): Expected '{expected_data}', got '{result_hit[0]}'"

    # Verify consistency between cache miss and hit results
    assert result_miss == result_hit, \
        f"Rank {rank}: Inconsistency between cache miss and hit results."

    print(f"Rank {rank}: Test passed. Cache miss and hit results are consistent.")

    cleanup()

if __name__ == "__main__":
    # Check if torch is available with distributed support
    if not torch.distributed.is_available():
        print("Distributed package not available. Skipping test.")
    else:
        world_size = 2
        # Spawn processes to simulate a distributed environment
        mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
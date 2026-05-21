import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def run_test(rank, world_size):
    # Initialize distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Use nccl backend for CUDA operations
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Prepare inputs
    # Rank 0 holds the object to broadcast, others hold dummy data
    if rank == 0:
        obj_list = ["data_from_rank_0", 123, 4.5]
    else:
        obj_list = [None, None, None]

    if torch.cuda.is_available():
        print(f"Rank {rank}: Starting CUDA Graph capture test with broadcast_object_list")
        
        # Attempt to capture CUDA Graph
        # This mimics the original test case structure where the API is called inside the graph context
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                # Adapted call site: torch.distributed.broadcast_object_list
                # Note: broadcast_object_list is a collective communication operation.
                # We are testing if it can be captured within a CUDA graph context.
                torch.distributed.broadcast_object_list(obj_list, src=0)
            
            # Replay graph
            graph.replay()
            
            # Verify results
            expected_list = ["data_from_rank_0", 123, 4.5]
            assert obj_list == expected_list, f"Rank {rank}: Mismatch. Got {obj_list}, expected {expected_list}"
            print(f"Rank {rank}: Test passed. Graph captured and replayed successfully.")
            
        except Exception as e:
            print(f"Rank {rank}: Test failed with error: {e}")
    else:
        print("CUDA not available, skipping test")

    dist.destroy_process_group()

if __name__ == "__main__":
    # Check for CUDA and Distributed availability
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
    elif not torch.distributed.is_available():
        print("Distributed package is not available. Exiting.")
    else:
        world_size = 2
        # Run the test in multiple processes
        mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)
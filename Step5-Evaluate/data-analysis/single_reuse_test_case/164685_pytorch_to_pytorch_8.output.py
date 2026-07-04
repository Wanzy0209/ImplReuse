import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    # Initialize the process group
    dist.init_process_group(
        backend="gloo", # Use gloo backend for CPU-based communication
        init_method=f"tcp://127.0.0.1:{12345}",
        rank=rank,
        world_size=world_size
    )

def cleanup():
    dist.destroy_process_group()

def run_worker(rank, world_size):
    setup(rank, world_size)

    if rank == 0:
        # --- Sender Logic ---
        # Replicate the data generation logic from the original bug report
        torch.manual_seed(19989)
        
        # Original types involved: int32, int64
        # Fix: Remove redundant torch.tensor() wrapper to avoid UserWarning
        arg_0 = torch.randn((), dtype=torch.int32).item()
        var_node_2 = -6 # dtype=int64
        var_node_3 = arg_0 # dtype=int32
        var_node_1 = var_node_2 * var_node_3 # dtype=int32
        var_node_5 = torch.full((), 1, dtype=torch.int64)
        var_node_4 = var_node_5.item() # dtype=int64
        
        # The operation that caused issues in the compiler
        var_node_0 = var_node_1 / var_node_4 

        # Prepare a list of objects to send
        # We include the specific types (int, int32 tensor, int64 tensor, float result)
        # to verify broadcast_object_list handles them correctly.
        objects_to_send = [
            var_node_0, 
            var_node_5, 
            torch.tensor(var_node_1, dtype=torch.int32)
        ]
        
        print(f"Rank {rank}: Broadcasting objects: {objects_to_send}")
        # Fix: Use broadcast_object_list instead of non-existent send_object_list
        dist.broadcast_object_list(objects_to_send, src=0)
        print(" broadcast_object_list success")

    elif rank == 1:
        # --- Receiver Logic (Testing the Similar API) ---
        # Initialize a list to receive objects
        objects_to_recv = [None, None, None]
        
        # Fix: Use broadcast_object_list instead of non-existent recv_object_list
        dist.broadcast_object_list(objects_to_recv, src=0)
        
        print(f"Rank {rank}: Received objects: {objects_to_recv}")
        
        # Assertions to verify data integrity
        assert isinstance(objects_to_recv[0], float), "Expected float for division result"
        assert isinstance(objects_to_recv[1], torch.Tensor), "Expected Tensor for var_node_5"
        assert objects_to_recv[1].dtype == torch.int64, "Expected int64 dtype"
        assert isinstance(objects_to_recv[2], torch.Tensor), "Expected Tensor for var_node_1"
        assert objects_to_recv[2].dtype == torch.int32, "Expected int32 dtype"
        
        print(" broadcast_object_list success and data verified")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    # Use multiprocessing to simulate distributed environment
    mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)
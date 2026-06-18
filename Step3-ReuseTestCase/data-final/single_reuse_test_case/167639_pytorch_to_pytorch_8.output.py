import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run(rank, world_size):
    setup(rank, world_size)
    
    # Ensure CUDA operations are on the correct device
    torch.cuda.set_device(rank)

    if rank == 0:
        # Receiver process
        print(f"Rank {rank}: Starting test.")
        
        # Create a dummy list to receive objects
        object_list = [None]
        
        # Synchronize to ensure sender is ready
        dist.barrier()

        if torch.cuda.is_available():
            print(f"Rank {rank}: Attempting to capture CUDA Graph with recv_object_list...")
            graph = torch.cuda.CUDAGraph()
            
            # Attempt to capture recv_object_list inside CUDA Graph
            # This tests the compatibility similar to the torch.compile issue
            try:
                with torch.cuda.graph(graph):
                    # Note: recv_object_list is a blocking call. 
                    # Capturing it in a graph is likely to fail or hang if not supported.
                    dist.recv_object_list(object_list, src=1)
                
                print(f"Rank {rank}: Graph capture successful.")
                
                # Replay the graph
                graph.replay()
                print(f"Rank {rank}: Graph replay successful.")
                print(f"Rank {rank}: Received object: {object_list[0]}")
                
            except Exception as e:
                print(f"Rank {rank}: Error during graph capture or replay: {e}")
        else:
            print("CUDA not available, skipping graph capture")

    else:
        # Sender process (Rank 1)
        dist.barrier()
        
        if torch.cuda.is_available():
            # Prepare data to send
            # Sending a CUDA tensor to ensure interaction with CUDA context
            data_to_send = [torch.randn(2, 2, device=torch.device(f'cuda:{rank}'))]
            
            print(f"Rank {rank}: Sending object...")
            dist.send_object_list(data_to_send, dst=0)
            print(f"Rank {rank}: Send complete.")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    # Check for CUDA availability before spawning processes
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        print("CUDA available with sufficient devices. Starting distributed test.")
        mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
    elif torch.cuda.is_available():
        print("CUDA available but requires at least 2 GPUs for this distributed test. Skipping.")
    else:
        print("CUDA not available. Skipping test.")
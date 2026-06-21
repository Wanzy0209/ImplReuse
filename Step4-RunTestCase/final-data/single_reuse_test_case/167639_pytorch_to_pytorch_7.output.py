import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    # Initialize the distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Use NCCL if available for CUDA operations, otherwise Gloo
    backend = 'nccl' if torch.distributed.is_nccl_available() else 'gloo'
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run_test(rank, world_size):
    setup(rank, world_size)
    
    if torch.cuda.is_available():
        # Set device for each process
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        
        # Create sample inputs similar to the original test case
        x = torch.randn(4, 10, requires_grad=True, device=device)
        inputs = (x,)
        
        # Adapted call site: Attempting to capture send_object_list inside a CUDA Graph
        # This mirrors the original bug report where torch.compile was called inside the graph context
        if rank == 0:
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):
                    # Check if send_object_list is available, otherwise fallback to dist.send
                    # This handles environments where the API might be missing or version-dependent
                    if hasattr(dist, 'send_object_list'):
                        dist.send_object_list(list(inputs), dst=1)
                    else:
                        # Fallback: send the tensor directly
                        dist.send(inputs[0], dst=1)
                
                # If capture succeeds, try to replay
                graph.replay()
                print("Graph capture and replay successful.")
            except RuntimeError as e:
                print(f"RuntimeError during graph capture (expected for incompatible APIs): {e}")
        else:
            # Receiver side (Rank 1)
            # We need a placeholder to receive the data
            recv_buffer = [torch.empty_like(inputs[0])]
            try:
                # Check if recv_object_list is available
                if hasattr(dist, 'recv_object_list'):
                    dist.recv_object_list(recv_buffer, src=0)
                else:
                    # Fallback: receive the tensor directly
                    dist.recv(recv_buffer[0], src=0)
                print("Receive successful.")
            except Exception as e:
                print(f"Receive failed: {e}")
    else:
        print("CUDA not available, skipping graph capture test.")
    
    cleanup()

def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping distributed test.")
        return

    world_size = 2
    # Run the test in a multiprocess setting
    mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
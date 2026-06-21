import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import sys

# Check for torch._dynamo availability
if not hasattr(torch, '_dynamo'):
    print("Skipping test: torch._dynamo is not available (requires PyTorch 2.0+)")
    sys.exit(0)

# Config from original bug report
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

if hasattr(torch, '_inductor'):
    torch._inductor.config.emulate_precision_casts = True

def setup():
    # Initialize the process group
    # Use 'gloo' for CPU compatibility or 'nccl' for CUDA
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend)

def cleanup():
    dist.destroy_process_group()

def run(rank, world_size):
    setup()
    
    # Scale down the size to make it runnable. 
    # Original size (5699097) causes OOM on standard hardware.
    # We use 100 to test the logic and API interaction without OOMing the test runner.
    size_dim = 100
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    # Create tensors on rank 0
    if rank == 0:
        arg0 = torch.rand([size_dim, 6, 1], dtype=torch.bfloat16, device=device)
        arg1 = torch.rand([size_dim, 6, 256], dtype=torch.bfloat16, device=device)
        arg2 = torch.rand([size_dim, 256, 1], dtype=torch.bfloat16, device=device)
    else:
        # Other ranks provide dummy data (will be overwritten)
        arg0 = torch.empty([size_dim, 6, 1], dtype=torch.bfloat16, device=device)
        arg1 = torch.empty([size_dim, 6, 256], dtype=torch.bfloat16, device=device)
        arg2 = torch.empty([size_dim, 256, 1], dtype=torch.bfloat16, device=device)

    # Define the function to test, adapted to use the similar API
    def foo(arg0, arg1, arg2):
        # Adaptation: Use torch.distributed.broadcast_object_list
        # instead of the math ops in the original bug report
        object_list = [arg0, arg1, arg2]
        dist.broadcast_object_list(object_list, src=0)
        return object_list

    # Run Eager
    out_eager = foo(arg0, arg1, arg2)
    if rank != 0:
        # Verify we received data from rank 0
        assert out_eager[0].abs().sum() > 0
        print(f"Rank {rank}: Eager Success! ")

    # Run Compiled
    # We test torch.compile (Original API) on the function using broadcast_object_list (Similar API)
    if hasattr(torch, 'compile'):
        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, arg1, arg2)
        
        if rank != 0:
            assert out_compiled[0].abs().sum() > 0
            print(f"Rank {rank}: Compile Success! ")
    else:
        print(f"Rank {rank}: torch.compile not available, skipping compiled execution.")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
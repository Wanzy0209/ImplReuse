import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

# Replicating the configuration from the original bug report
# to ensure the environment is set up similarly for compilation.
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initialize process group using gloo backend for CPU compatibility
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run_test(rank, world_size):
    setup(rank, world_size)

    # Create tensors similar to the original bug report
    # Using CPU to ensure the test runs in environments without CUDA
    device = 'cpu' 
    
    arg0 = torch.rand([27, 26, 62, 122], dtype=torch.float32, device=device)
    arg1 = torch.rand([27, 26, 124, 122], dtype=torch.float32, device=device)
    arg2 = torch.rand([27, 26, 124, 122], dtype=torch.float32, device=device)
    arg3 = torch.rand([27, 26, 124, 122], dtype=torch.float32, device=device)
    arg4 = torch.rand([27, 26, 248, 122], dtype=torch.float32, device=device)
    arg5 = torch.rand([27, 26, 248, 122], dtype=torch.float32, device=device)

    if rank == 0:
        # Adapted function replacing flex_attention with send_object_list
        def foo(arg0, arg1, arg2):
            # Original call site: t3 = flex_attention(t0, t1, t2)
            # Replaced with: dist.send_object_list(...)
            # We send the input tensors as objects to verify the API handles them.
            dist.send_object_list([arg0, arg1, arg2], dst=1)

        # Compile the function to test for eager/compile divergence
        # similar to the original bug report.
        compiled_foo = torch.compile(foo)
        
        # Execute the compiled function
        compiled_foo(arg0, arg1, arg2)
        print("Rank 0: Send operation completed successfully under torch.compile.")

    elif rank == 1:
        # Receiver logic to verify the data
        recv_list = [None, None, None]
        dist.recv_object_list(recv_list, src=0)
        
        # Verify received data matches sent data
        assert torch.allclose(recv_list[0], arg0), "Mismatch in arg0"
        assert torch.allclose(recv_list[1], arg1), "Mismatch in arg1"
        assert torch.allclose(recv_list[2], arg2), "Mismatch in arg2"
        print("Rank 1: Receive operation completed and data verified.")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    # Use spawn to run the distributed test
    mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)
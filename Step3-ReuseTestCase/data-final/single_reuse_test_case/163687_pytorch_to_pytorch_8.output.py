import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

# Replicating the configuration from the original bug report
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run(rank, world_size):
    setup(rank, world_size)

    if rank == 0:
        # Sender: Create tensors matching the shapes from the original test case
        arg0 = torch.rand([27, 26, 62, 122], dtype=torch.float32, device='cuda')
        arg1 = torch.rand([27, 26, 124, 122], dtype=torch.float32, device='cuda')
        arg2 = torch.rand([27, 26, 124, 122], dtype=torch.float32, device='cuda')
        arg3 = torch.rand([27, 26, 124, 122], dtype=torch.float32, device='cuda')
        arg4 = torch.rand([27, 26, 248, 122], dtype=torch.float32, device='cuda')
        arg5 = torch.rand([27, 26, 248, 122], dtype=torch.float32, device='cuda')
        arg6 = torch.rand([27, 26, 31, 122], dtype=torch.float32, device='cuda')
        arg7 = torch.rand([27, 26, 124, 122], dtype=torch.float32, device='cuda')
        arg8 = torch.rand([27, 26, 31, 122], dtype=torch.float32, device='cuda')
        arg9 = torch.rand([27, 26, 124, 122], dtype=torch.float32, device='cuda')
        arg10 = torch.rand([27, 26, 124, 122], dtype=torch.float32, device='cuda')
        
        tensors = [arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10]
        
        # Send objects
        dist.send_object_list(tensors, dst=1)
        print("Rank 0: Sent tensors.")

    elif rank == 1:
        # Receiver: Adapt the test to use torch.distributed.recv_object_list
        # We test both eager and compiled modes to check for divergence (OOM or correctness)
        
        received = [None] * 11
        
        def recv_fn():
            # Call the similar API
            dist.recv_object_list(received, src=0)
            return received

        # 1. Test Eager Mode
        print("Rank 1: Testing Eager mode...")
        recv_fn()
        assert all(t is not None for t in received), "Eager receive failed: tensors are None"
        assert all(isinstance(t, torch.Tensor) for t in received), "Eager receive failed: items are not tensors"
        print("Rank 1: Eager mode successful.")

        # Reset for compile test
        received = [None] * 11
        
        # 2. Test Compile Mode
        # The original bug reported OOM in compile. We check if recv_object_list behaves differently.
        print("Rank 1: Testing Compile mode...")
        try:
            compiled_recv = torch.compile(recv_fn)
            compiled_recv()
            
            assert all(t is not None for t in received), "Compile receive failed: tensors are None"
            assert all(isinstance(t, torch.Tensor) for t in received), "Compile receive failed: items are not tensors"
            print("Rank 1: Compile mode successful.")
        except Exception as e:
            print(f"Rank 1: Compile mode failed with error: {e}")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. This test requires CUDA.")
    else:
        mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
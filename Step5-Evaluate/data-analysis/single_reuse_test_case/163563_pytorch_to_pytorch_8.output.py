import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

# Bug specific configurations
# Guard against older PyTorch versions where _dynamo might not exist
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
if hasattr(torch, '_inductor'):
    torch._inductor.config.emulate_precision_casts = True

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def run_test(rank, world_size):
    setup(rank, world_size)

    if rank == 0:
        # Rank 0 acts as the sender
        # Create tensors matching the bug report's characteristics
        arg0 = torch.rand([5699097, 6, 1], dtype=torch.bfloat16, device='cuda')
        arg1 = torch.rand([5699097, 6, 256], dtype=torch.bfloat16, device='cuda')
        arg2 = torch.rand([5699097, 256, 1], dtype=torch.bfloat16, device='cuda')

        # Perform the operations from the bug report to generate the payload
        t0 = arg0
        t1 = torch.sigmoid(t0)
        t2 = arg1
        t3 = torch.sigmoid(t2)
        t4 = arg2
        t5 = torch.exp(t4)
        t6 = torch.baddbmm(t1, t3, t5)
        t7 = t6.reshape((193, 386, 459))

        # Send the result
        dist.send_object_list([t7], dst=1)
        print("Rank 0: Sent data successfully")

    else:
        # Rank 1 acts as the receiver
        # Define the function to be tested (similar to 'foo' in the original bug)
        def recv_fn():
            obj_list = [None]
            dist.recv_object_list(obj_list, src=0)
            return obj_list[0]

        # Test Eager
        print("Rank 1: Testing Eager...")
        out_eager = recv_fn()
        print(f"Rank 1: Eager Success! Shape: {out_eager.shape}")

        # Test Compiled (checking for OOM divergence)
        # Check if torch.compile is available to support older PyTorch versions
        if hasattr(torch, 'compile'):
            print("Rank 1: Testing Compiled...")
            compiled_recv = torch.compile(recv_fn, fullgraph=True, dynamic=True)
            out_compiled = compiled_recv()
            print(f"Rank 1: Compile Success! Shape: {out_compiled.shape}")

            # Basic assertion
            assert out_eager.shape == out_compiled.shape
        else:
            print("Rank 1: torch.compile is not available. Skipping compiled test.")

    cleanup()

if __name__ == "__main__":
    # Check for CUDA availability as the bug is specific to CUDA
    if not torch.cuda.is_available():
        print("CUDA is not available. This test requires CUDA.")
    else:
        world_size = 2
        mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)
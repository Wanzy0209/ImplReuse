import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def foo(arg0, arg1):
    # Original Bug Reproduction Logic
    # This function contains the logic that triggered the assert_size_stride bug
    t0 = arg0 # size=(4, 503, 64, 504), stride=(16224768, 32256, 504, 1), dtype=float32, device=cuda
    t1 = t0.mean(dim=0) # size=(503, 64, 504), stride=(32192, 64, 1), dtype=float32, device=cuda
    t2 = torch.nn.functional.relu(t1) # size=(503, 64, 504), stride=(32192, 64, 1), dtype=float32, device=cuda
    t3 = arg1 # size=(5, 16, 1, 64), stride=(1024, 64, 64, 1), dtype=float32, device=cuda
    t4 = t3.sum(dim=0) # size=(16, 1, 64), stride=(1024, 1, 64), dtype=float32, device=cuda
    t5 = t4.transpose(2, 1) # size=(16, 64, 1), stride=(1024, 64, 1), dtype=float32, device=cuda
    t6 = torch.nn.functional.conv1d(t2, t5, stride=1, padding=0) # size=(503, 16, 504), stride=(8064, 504, 1), dtype=float32, device=cuda
    output = t6  # output tensor
    return output

def run_test(rank, world_size):
    # Initialize the distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Leverage Similar API: torch.distributed.get_global_rank
    # We use this API to verify the rank mapping, mirroring the validation pattern
    # found in the original bug report (assert_size_stride).
    global_rank = dist.get_global_rank(dist.group.WORLD, rank)
    assert global_rank == rank, f"Global rank mismatch: expected {rank}, got {global_rank}"

    # Run the original bug reproduction logic on the primary rank
    if rank == 0:
        if torch.cuda.is_available():
            # Check for availability of torch._dynamo and torch._inductor
            # to avoid AttributeError in older PyTorch versions
            use_dynamo = hasattr(torch, '_dynamo') and hasattr(torch, '_inductor')
            
            if use_dynamo:
                # Configs from the original bug report
                torch._dynamo.config.capture_scalar_outputs = True
                torch._dynamo.config.capture_dynamic_output_shape_ops = True
                torch._inductor.config.emulate_precision_casts = True

            arg0 = torch.rand([4, 503, 64, 504], dtype=torch.float32, device='cuda', requires_grad=True)
            arg1 = torch.rand([5, 16, 1, 64], dtype=torch.float32, device='cuda', requires_grad=True)

            # Eager execution
            out_eager = foo(arg0, arg1)
            out_eager.sum().backward()
            print('Eager Success! ')

            # Compiled execution
            if use_dynamo and hasattr(torch, 'compile'):
                compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
                out_compiled = compiled_foo(arg0, arg1)
                out_compiled.sum().backward()
                print('Compile Success! ')
            else:
                print("Skipping compiled execution (torch._dynamo or torch.compile not available).")
        else:
            print("CUDA not available, skipping tensor operations.")

    dist.destroy_process_group()

if __name__ == '__main__':
    # Use a single process for minimal reproducibility while satisfying distributed API requirements
    world_size = 1
    mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)
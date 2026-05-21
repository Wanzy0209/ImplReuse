import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initialize process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def foo(arg0, arg1, arg2):
    t0 = arg0
    t1 = torch.sigmoid(t0)
    t2 = arg1
    t3 = torch.sigmoid(t2)
    t4 = arg2
    t5 = torch.exp(t4)
    t6 = torch.baddbmm(t1, t3, t5)
    t7 = t6.reshape((193, 386, 459))
    
    # Adaptation: Use torch.distributed.reduce on the output tensor
    # This verifies the similar API in the context of the original computation
    dist.reduce(t7, dst=0)
    
    return t7

def run(rank, world_size):
    setup(rank, world_size)
    
    # Create tensors on CUDA if available, otherwise CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Note: The original bug report used very large tensors that caused OOM.
    # We keep the sizes to verify the behavior, but ensure the test environment 
    # has sufficient memory.
    arg0 = torch.rand([5699097, 6, 1], dtype=torch.bfloat16, device=device, requires_grad=True)
    arg1 = torch.rand([5699097, 6, 256], dtype=torch.bfloat16, device=device, requires_grad=True)
    arg2 = torch.rand([5699097, 256, 1], dtype=torch.bfloat16, device=device, requires_grad=True)

    # Eager execution
    try:
        out_eager = foo(arg0, arg1, arg2)
        if rank == 0:
            out_eager.sum().backward()
            print(f'Rank {rank}: Eager Success! ')
    except Exception as e:
        if rank == 0:
            print(f'Rank {rank}: Eager Failed!  {e}')

    # Compiled execution
    # We compile the function which now includes torch.distributed.reduce
    try:
        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, arg1, arg2)
        if rank == 0:
            out_compiled.sum().backward()
            print(f'Rank {rank}: Compile Success! ')
    except Exception as e:
        if rank == 0:
            print(f'Rank {rank}: Compile Failed!  {e}')

    cleanup()

if __name__ == '__main__':
    world_size = 2
    # Use spawn to launch processes for distributed testing
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
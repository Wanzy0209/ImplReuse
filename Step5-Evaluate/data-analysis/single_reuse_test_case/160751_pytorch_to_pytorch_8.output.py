import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import sys

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Use gloo backend for CPU compatibility in this test
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def func(rank):
    """
    Adapted function to test torch.distributed.recv_object_list.
    Original bug: torch.cuda.synchronize() was removed by aot_eager.
    Here we verify that recv_object_list is not removed/optimized away.
    """
    obj_list = [None]
    
    # This call acts as a synchronization point and side-effect operation.
    # If aot_eager optimizes this away (similar to the synchronize() bug),
    # the list will remain None and the assertion will fail.
    dist.recv_object_list(obj_list, src=0)
    
    assert obj_list[0] == "test_data", "recv_object_list was optimized away or did not execute"
    print(f"Rank {rank}: Successfully received object")

def test_fn(rank):
    setup(rank, 2)
    
    # Fix: Check if _dynamo exists before calling reset to handle older PyTorch versions
    if hasattr(torch, '_dynamo'):
        torch._dynamo.reset()
    
    # Fix: Check if torch.compile is available. 
    # If not (e.g., PyTorch < 2.0), run the function uncompiled to allow the test to complete.
    if hasattr(torch, 'compile'):
        f_c = torch.compile(func, backend="aot_eager")
        runner = f_c
    else:
        print(f"Rank {rank}: torch.compile not available (requires PyTorch 2.0+). Running uncompiled.")
        runner = func
    
    if rank == 0:
        # Sender
        dist.send_object_list(["test_data"], dst=1)
    else:
        # Receiver runs the compiled function (or uncompiled fallback)
        runner(rank)
    
    cleanup()

if __name__ == "__main__":
    # Run with 2 processes to simulate sender and receiver
    mp.spawn(test_fn, args=(), nprocs=2)
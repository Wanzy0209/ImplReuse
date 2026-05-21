import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def test_irecv(rank, world_size):
    # Initialize the distributed process group
    # Using 'gloo' backend for CPU tensors
    dist.init_process_group(
        backend='gloo',
        init_method=f'tcp://127.0.0.1:29500',
        rank=rank,
        world_size=world_size
    )

    # Create the NestedTensor as described in the bug report
    a = torch.randn(3)
    b = torch.randn(5)
    nt = torch.nested.nested_tensor([a, b], layout=torch.jagged)

    # Adapt the original call site (nt.share_memory_()) to the similar API (dist.irecv)
    # We verify if irecv can handle a NestedTensor as the buffer without crashing
    try:
        if rank == 0:
            print(f"Rank {rank}: Attempting irecv with NestedTensor...")
            # Pass the NestedTensor to irecv
            work = dist.irecv(nt, src=1)
            print(f"Rank {rank}: irecv call succeeded (returned {work}).")
        else:
            # Rank 1 is the source, but we only test the receive call site here
            pass
    except Exception as e:
        print(f"Rank {rank}: Exception caught: {type(e).__name__}: {e}")

    dist.destroy_process_group()

if __name__ == "__main__":
    # Spawn 2 processes to allow for distributed communication testing
    world_size = 2
    mp.spawn(test_irecv, args=(world_size,), nprocs=world_size, join=True)
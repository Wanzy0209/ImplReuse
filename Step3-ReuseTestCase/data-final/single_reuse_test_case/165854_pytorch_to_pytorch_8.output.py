import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os


def run_with_size(compiled_recv, size, rank, src_rank=0):
    """Run recv_object_list with a specific list size, creating a buffer sized by size."""
    # Create captured buffer that depends on dynamic size
    # Note: recv_object_list modifies the list in place.
    object_list = [None] * size

    print(f"  Rank {rank}: Receiving with size={size}, object_list length={len(object_list)}")

    # Run the receive operation
    compiled_recv(object_list, src=src_rank)

    # Verify
    assert len(object_list) == size
    assert all(isinstance(obj, torch.Tensor) for obj in object_list)
    print(f"  Rank {rank}:  Completed receive for size={size}")


def main_worker(rank, world_size):
    # Initialize process group
    dist.init_process_group(
        backend='gloo', # Use gloo for CPU compatibility
        init_method='tcp://127.0.0.1:29500',
        rank=rank,
        world_size=world_size
    )

    # Test with different list sizes - this makes size a dynamic dimension
    # and the captured buffer (object_list) changes size with size
    sizes = [4, 8, 4, 16, 4]

    if rank == 0:
        # Sender logic
        for size in sizes:
            data = [torch.randn(2, 2) for _ in range(size)]
            dist.send_object_list(data, dst=1)
    else:
        # Receiver logic
        # Define the function to be compiled
        def recv_logic(obj_list, src):
            dist.recv_object_list(obj_list, src=src)

        # Compile the function (mimicking the original bug's setup)
        # Note: Compiling distributed ops is experimental.
        try:
            compiled_recv = torch.compile(recv_logic)
        except Exception as e:
            print(f"Compilation failed (expected in some versions): {e}")
            compiled_recv = recv_logic

        print(f"Rank {rank}: Running recv_object_list with dynamic sizes")
        print(f"Testing sizes: {sizes}\n")

        for iteration, size in enumerate(sizes, start=1):
            print(f"Iteration {iteration}:")
            run_with_size(compiled_recv, size, rank, src_rank=0)


if __name__ == "__main__":
    world_size = 2
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os


def run_with_data_size(rank, world_size, size):
    """Run gather_object with a specific data size, creating an object sized by 'size'."""

    # Create object that depends on dynamic 'size'
    # Each rank creates a unique list based on its rank and the current size
    local_obj = [rank * size + i for i in range(size)]

    # Prepare the gather list (buffer) on the destination
    if rank == 0:
        gather_list = [None] * world_size
    else:
        gather_list = None

    print(f"  Rank {rank}: Running with size={size}, local_obj={local_obj}")

    # Run the operation
    dist.gather_object(local_obj, gather_list, dst=0)

    # Verification on rank 0
    if rank == 0:
        assert len(gather_list) == world_size
        for r in range(world_size):
            expected = [r * size + i for i in range(size)]
            assert gather_list[r] == expected
        print(f"   Rank 0: Verified gather for size={size}")


def main():
    world_size = 2
    # Setup environment variables for multiprocessing
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Test with different data sizes - this makes 'size' a dynamic parameter
    # and the object being gathered changes size with 'size'
    data_sizes = [4, 8, 4, 16, 4]

    def worker(rank):
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

        print(f"Running gather_object with dynamic data sizes")
        print(f"Testing sizes: {data_sizes}\n")

        for iteration, size in enumerate(data_sizes, start=1):
            print(f"Iteration {iteration}:")
            run_with_data_size(rank, world_size, size)

        dist.destroy_process_group()

    mp.spawn(worker, args=(), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
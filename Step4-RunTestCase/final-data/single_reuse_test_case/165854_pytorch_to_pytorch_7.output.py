import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os


def run_with_size(rank, world_size, H, device, dtype):
    """Run send_object_list with a specific size, creating a buffer sized by H."""
    
    # Create captured buffer that depends on dynamic H
    # This mimics the 'head_scale' tensor from the original bug report
    dynamic_buffer = torch.randn(H, device=device, dtype=dtype)

    object_list = [dynamic_buffer, f"metadata_size_{H}"]

    if rank == 0:
        print(f"  Running with H={H}, dynamic_buffer.shape={dynamic_buffer.shape}")
        # Send the list containing the dynamic buffer
        dist.send_object_list(object_list, dst=1, device=device)
    else:
        # Receive the list
        # The receiver must provide a list of the correct length
        recv_list = [None, None]
        dist.recv_object_list(recv_list, src=0, device=device)

        # Verify the received tensor shape matches the dynamic H
        assert recv_list[0].shape == (H,), f"Shape mismatch: expected ({H},), got {recv_list[0].shape}"
        assert recv_list[1] == object_list[1]
        print(f"   Verified reception for H={H}")


def setup_and_run(rank, world_size, device, dtype, sizes):
    # Initialize process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    print(f"Running send_object_list with dynamic sizes on rank {rank}, device={device}")
    print(f"Testing sizes: {sizes}\n")

    for iteration, H in enumerate(sizes, start=1):
        if rank == 0:
            print(f"Iteration {iteration}:")
        # Synchronize to ensure matching iterations between sender and receiver
        dist.barrier()
        run_with_size(rank, world_size, H, device, dtype)

    dist.destroy_process_group()


def main():
    device = "cpu" # Using CPU for broader compatibility in this test
    dtype = torch.float16
    torch.manual_seed(0)

    # Test with different sizes - this makes H a dynamic dimension
    # and the captured buffer (dynamic_buffer) changes size with H
    sizes = [4, 8, 4, 16, 4]

    # Spawn 2 processes (sender and receiver)
    world_size = 2
    mp.spawn(setup_and_run, args=(world_size, device, dtype, sizes), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
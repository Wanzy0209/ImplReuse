import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def test_send_object_list(rank, world_size):
    """
    Test case for torch.distributed.send_object_list on XPU.
    Adapted from the original torch.compile segfault issue to verify
    similar API stability with XPU tensors.
    """
    # Setup environment variables for distributed communication
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    # Check for XPU availability
    if not torch.xpu.is_available():
        print(f"Rank {rank}: XPU is not available. Skipping test.")
        return

    # Initialize the process group
    # 'ccl' (oneCCL) is the backend typically used for Intel XPU
    try:
        dist.init_process_group(backend="ccl", rank=rank, world_size=world_size)
    except Exception as e:
        print(f"Rank {rank}: Error initializing process group: {e}")
        return

    # Create tensors on XPU (mirroring the original bug report's data setup)
    x = torch.randn(128).to("xpu")
    y = torch.randn(128).to("xpu")
    z = torch.randn(128).to("xpu")

    if rank == 0:
        # Adaptation: Replace torch.compile call with torch.distributed.send_object_list
        # We attempt to send the list of XPU tensors to rank 1
        object_list = [x, y, z]
        try:
            dist.send_object_list(object_list, dst=1)
            print("Rank 0: send_object_list passed")
        except Exception as e:
            print(f"Rank 0: send_object_list failed with error: {e}")
    elif rank == 1:
        # Receiver side to validate the send operation completes without segfault
        object_list = [None, None, None]
        try:
            dist.recv_object_list(object_list, src=0)
            print("Rank 1: recv_object_list passed")
            
            # Verify data integrity
            assert len(object_list) == 3
            assert all(isinstance(t, torch.Tensor) for t in object_list)
            assert object_list[0].shape == (128,)
            print("Rank 1: Data validation passed")
        except Exception as e:
            print(f"Rank 1: recv_object_list failed with error: {e}")

    dist.destroy_process_group()

if __name__ == "__main__":
    # Spawn 2 processes to simulate a distributed environment
    # This allows the test to be run as a standalone script
    world_size = 2
    mp.spawn(test_send_object_list, args=(world_size,), nprocs=world_size, join=True)
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def test_broadcast_object_list(rank, world_size):
    """
    Test case for torch.distributed.broadcast_object_list.
    Adapted from the context of the FlexAttention bug report to verify
    distributed communication with similar tensor characteristics.
    """
    # Setup distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group using NCCL (original bug was CUDA specific)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Prepare data similar to the original bug report
    # Original: q = torch.randn([2, 32, 4096, 128], dtype=torch.bfloat16, ...)
    if rank == 0:
        # Create a list of objects to broadcast
        q = torch.randn([2, 32, 4096, 128], dtype=torch.bfloat16, device="cuda")
        k = torch.randn([2, 8, 4096, 128], dtype=torch.bfloat16, device="cuda")
        object_list = [q, k, "test_metadata"]
    else:
        object_list = [None, None, None]

    # Call the similar API: torch.distributed.broadcast_object_list
    dist.broadcast_object_list(object_list, src=0)

    # Verify results on non-source ranks
    if rank != 0:
        assert isinstance(object_list[0], torch.Tensor), "Failed to receive tensor q"
        assert object_list[0].shape == (2, 32, 4096, 128), "Shape mismatch for q"
        assert object_list[0].dtype == torch.bfloat16, "Dtype mismatch for q"
        
        assert isinstance(object_list[1], torch.Tensor), "Failed to receive tensor k"
        assert object_list[1].shape == (2, 8, 4096, 128), "Shape mismatch for k"
        
        assert object_list[2] == "test_metadata", "Failed to receive string metadata"
        print(f"Rank {rank} verification passed.")

    dist.destroy_process_group()

if __name__ == "__main__":
    # Check for CUDA availability as the original bug was specific to NVIDIA hardware
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test.")
    else:
        world_size = 2
        mp.spawn(test_broadcast_object_list, args=(world_size,), nprocs=world_size, join=True)
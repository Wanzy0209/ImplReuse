import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def test_gather_object(rank, world_size):
    """
    Test case for torch.distributed.gather_object.
    Adapted from the context of the FlexAttention bug report to verify
    the similar API (torch.distributed.gather_object).
    """
    
    # Setup distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Create input objects. 
    # Mimicking the tensor shapes from the original bug report (bfloat16, specific dims).
    # gather_object handles picklable objects, so we wrap tensors in a dict.
    input_obj = {
        "rank": rank,
        "tensor": torch.randn([2, 32, 128], dtype=torch.bfloat16)
    }

    # Prepare output list on the destination rank (rank 0)
    if rank == 0:
        gathered_objects = [None] * world_size
    else:
        gathered_objects = None

    # Call the API: torch.distributed.gather_object
    dist.gather_object(
        obj=input_obj,
        object_gather_list=gathered_objects,
        dst=0
    )

    # Verification
    if rank == 0:
        assert len(gathered_objects) == world_size, "Gathered list length mismatch"
        for i in range(world_size):
            obj = gathered_objects[i]
            assert obj["rank"] == i, f"Rank mismatch at index {i}"
            assert obj["tensor"].shape == (2, 32, 128), f"Tensor shape mismatch for rank {i}"
            assert obj["tensor"].dtype == torch.bfloat16, f"Tensor dtype mismatch for rank {i}"
        print("Test passed: torch.distributed.gather_object verified successfully.")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(test_gather_object, args=(world_size,), nprocs=world_size, join=True)
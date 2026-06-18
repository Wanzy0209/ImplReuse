import torch
import torch.distributed as dist
import os

def main():
    # Setup based on the bug report environment
    # These environment variables are typically set by the launcher (e.g., torchrun)
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)
    
    # The bug report indicates a segfault occurs here on large NVL72 clusters (>40 GPUs)
    # during the NCCL topology search.
    dist.init_process_group(
        backend='nccl',
        device_id=local_rank,
    )
    
    # Test the similar API: torch.distributed.broadcast_object_list
    # This function requires an initialized process group.
    # We replace the original 'dist.barrier()' with this call to verify functionality
    # in the context of the large-scale setup.
    
    obj_list = [None]
    if rank == 0:
        # Create a picklable object containing a tensor on the specific device
        obj_list = [{"tensor": torch.randn(2, 2).to(device), "value": 42}]
    
    # Broadcast the object list from rank 0 to all other ranks
    dist.broadcast_object_list(obj_list, src=0, device=torch.device(device))
    
    # Assertions to verify the broadcast worked correctly
    if rank != 0:
        assert obj_list[0] is not None
        assert obj_list[0]["value"] == 42
        assert obj_list[0]["tensor"].device == torch.device(device)
        assert obj_list[0]["tensor"].shape == (2, 2)
    
    print(f"Rank {rank} successfully executed broadcast_object_list.")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
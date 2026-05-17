import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    """
    Initialize the distributed environment.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    """
    Destroy the process group.
    """
    dist.destroy_process_group()

def run_test(rank, world_size):
    """
    Test torch.distributed.gather_object with tensors on the 'meta' device.
    
    This test is adapted from the bug report which highlights issues with 
    'meta' device tensors appearing in custom backend operations. 
    We verify if gather_object can handle meta tensors correctly via pickling.
    """
    setup(rank, world_size)

    # Create a tensor on the 'meta' device.
    # The bug report mentions: "Got: meta" causing a RuntimeError.
    # We test if gather_object can serialize/deserialize this.
    meta_tensor = torch.empty(2, 3, device="meta")
    
    # Add some custom metadata to the object to verify integrity
    # (since meta tensors have no actual data values to check)
    custom_obj = {"id": rank, "tensor": meta_tensor}

    if rank == 0:
        gather_list = [None for _ in range(world_size)]
    else:
        gather_list = None

    # Perform the gather operation
    try:
        dist.gather_object(custom_obj, gather_list, dst=0)
    except Exception as e:
        print(f"Rank {rank} failed during gather_object: {e}")
        cleanup()
        return

    # Verification on the destination rank
    if rank == 0:
        print("Gather successful. Verifying results...")
        assert len(gather_list) == world_size, "Gather list length mismatch"
        
        for i, obj in enumerate(gather_list):
            assert obj["id"] == i, f"Object ID mismatch at index {i}"
            t = obj["tensor"]
            # Verify the tensor properties are preserved
            assert t.device.type == 'meta', f"Expected meta device, got {t.device}"
            assert t.shape == (2, 3), f"Shape mismatch at index {i}"
            print(f"Rank 0 verified object from Rank {i}: device={t.device}, shape={t.shape}")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    # Use spawn to launch processes for distributed testing
    mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)
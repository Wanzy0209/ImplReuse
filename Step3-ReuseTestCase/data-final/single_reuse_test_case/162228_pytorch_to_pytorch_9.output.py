import torch
import torch.distributed as dist
import os

def setup():
    # Initialize the process group
    # Using 'gloo' backend as it is widely supported for CPU and basic testing
    dist.init_process_group(backend="gloo")

def cleanup():
    dist.destroy_process_group()

@torch.compile()
def test_broadcast(object_list, src=0):
    # Adapted call site: replacing flex_attention with broadcast_object_list
    # Note: broadcast_object_list modifies the list in-place and returns None in some versions,
    # or returns the list. We rely on the in-place modification for the check.
    dist.broadcast_object_list(object_list, src=src)
    return object_list

if __name__ == "__main__":
    # Check if distributed environment variables are set
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Only run if distributed setup is detected (e.g., via torchrun)
    if world_size > 1:
        setup()
        
        # Set seed for reproducibility across ranks
        torch.manual_seed(42)
        
        # Prepare data
        # Rank 0 holds the data to broadcast
        if rank == 0:
            # Create a tensor and wrap it in a dict/list to simulate an object
            tensor_to_send = torch.randn(2, 2)
            data = [{"id": 1, "tensor": tensor_to_send}]
            print(f"Rank {rank} broadcasting: {data}")
        else:
            data = [None]

        # Execute the compiled function
        # This mimics the structure of the original bug report where a compiled function
        # calls the target API.
        result = test_broadcast(data, src=0)

        # Verify
        # Since we set the seed, we can verify the tensor content on non-src ranks
        if rank != 0:
            assert result is not None, "Rank {} received None".format(rank)
            assert len(result) == 1, "Rank {} received list of incorrect length".format(rank)
            assert "tensor" in result[0], "Rank {} received object without 'tensor' key".format(rank)
            
            # Regenerate expected tensor based on seed
            expected_tensor = torch.randn(2, 2)
            assert torch.allclose(result[0]["tensor"], expected_tensor), \
                "Rank {} received incorrect tensor values".format(rank)
            print(f"Rank {rank} successfully received and verified data.")
        else:
            # Rank 0 checks that the list wasn't corrupted (though broadcast_object_list is usually in-place)
            assert result[0]["tensor"] is not None
            print(f"Rank {rank} finished.")

        cleanup()
    else:
        print("This test requires a distributed environment (e.g., torchrun --nproc_per_node=2).")
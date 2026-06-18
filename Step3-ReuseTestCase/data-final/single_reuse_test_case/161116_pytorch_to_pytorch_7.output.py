import torch
import torch.distributed as dist
import os

def main():
    # Retrieve environment variables set by the distributed launcher
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    # Set the device for the current process
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)
    
    # Initialize the process group
    # Note: The original bug (Issue 161116) reports a segfault occurring here 
    # on large NVL72 clusters (>10 trays).
    dist.init_process_group(backend='nccl',
                            device_id=local_rank,
                            )
    
    # Adapted test case for torch.distributed.scatter_object_list
    # Prepare output list to store the received object
    scatter_object_output_list = [None]
    
    # Prepare input list on the source rank (rank 0)
    if rank == 0:
        scatter_object_input_list = [f"object_{i}" for i in range(world_size)]
    else:
        scatter_object_input_list = None
        
    # Perform the scatter operation
    dist.scatter_object_list(scatter_object_output_list, 
                             scatter_object_input_list, 
                             src=0)
    
    # Verify that the object received matches the expected object for this rank
    expected_object = f"object_{rank}"
    assert scatter_object_output_list[0] == expected_object, \
        f"Rank {rank} failed verification. Expected {expected_object}, got {scatter_object_output_list[0]}"
    
    print(f"Rank {rank} successfully received: {scatter_object_output_list[0]}")
    
    # Clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
import torch
import torch.distributed as dist
import os

def main():
    # Retrieve environment variables set by the launcher (e.g., torchrun)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # Fix: Ensure required environment variables for the default 'env://' rendezvous are set.
    # This prevents the ValueError when running the script without a launcher (e.g., torchrun).
    if "RANK" not in os.environ:
        os.environ["RANK"] = str(rank)
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = str(world_size)
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"

    # Set the device for the current process
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Initialize the process group
    # Note: The original bug report indicates a potential segfault here on large clusters (>40 GPUs).
    dist.init_process_group(backend='nccl')

    # Test torch.distributed.send_object_list
    # This API requires a sender and a receiver. We use rank 0 and rank 1.
    if world_size >= 2:
        if rank == 0:
            # Prepare a list of picklable objects to send
            obj_list = ["test_string", 123, {"key": "value"}]
            # Send the list to rank 1
            dist.send_object_list(obj_list, dst=1)
        elif rank == 1:
            # Prepare a list to receive the objects
            # The list must be pre-allocated with the correct size
            recv_list = [None] * 3
            # Receive the list from rank 0
            dist.recv_object_list(recv_list, src=0)

            # Assertions to verify data integrity
            assert recv_list[0] == "test_string", "String mismatch"
            assert recv_list[1] == 123, "Integer mismatch"
            assert recv_list[2] == {"key": "value"}, "Dictionary mismatch"
            
            print(f"Rank {rank} successfully received and verified objects.")

    # Synchronize all processes
    dist.barrier()

if __name__ == "__main__":
    main()
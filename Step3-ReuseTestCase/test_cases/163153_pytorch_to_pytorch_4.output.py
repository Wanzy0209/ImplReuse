import torch
import torch.distributed as dist
import os

def verify_min_gpu_count(min_gpus: int = 2) -> bool:
    """Verification that we have at least 2 gpus to run dist examples"""
    has_gpu = torch.accelerator.is_available()
    gpu_count = torch.accelerator.device_count()
    return has_gpu and gpu_count >= min_gpus

def main():
    _min_gpu_count = 2
    if not verify_min_gpu_count(min_gpus=_min_gpu_count):
        print(f"Unable to locate sufficient {_min_gpu_count} gpus to run this example. Exiting.")
        return

    rank = int(os.environ["LOCAL_RANK"])
    if torch.accelerator.is_available():
        device_type = torch.accelerator.current_accelerator()
        device = torch.device(f"{device_type}:{rank}")
        torch.accelerator.set_device_index(rank)
        print(f"Running on rank {rank} on device {device}")
    else:
        device = torch.device("cpu")
        print(f"Running on device {device}")

    backend = dist.get_default_backend_for_device(device)
    dist.init_process_group(backend=backend, device_id=device)

    # Test torch.distributed.send_object_list
    # This API sends picklable objects synchronously to a destination rank.
    
    if rank == 0:
        # Prepare a list of objects to send
        # Including a dict, a tensor, and a string to test pickling capability
        objects_to_send = [
            {"model_config": "fsdp2_test", "layers": 10},
            torch.randn(2, 2).to(device),
            "checkpoint_data"
        ]
        
        dst_rank = 1
        print(f"Rank {rank} sending object list to rank {dst_rank}...")
        
        # Call the similar API: torch.distributed.send_object_list
        dist.send_object_list(objects_to_send, dst=dst_rank)
        
        print(f"Rank {rank} send complete.")

    elif rank == 1:
        src_rank = 0
        # Prepare a list to receive data. Size must match sender.
        received_objects = [None] * 3
        
        print(f"Rank {rank} waiting to receive object list from rank {src_rank}...")
        
        # Use recv_object_list to verify the send operation
        dist.recv_object_list(received_objects, src=src_rank)
        
        print(f"Rank {rank} received objects.")
        
        # Assertions to verify data integrity
        assert isinstance(received_objects[0], dict), "First object is not a dict"
        assert received_objects[0]["model_config"] == "fsdp2_test", "Dict content mismatch"
        
        assert torch.is_tensor(received_objects[1]), "Second object is not a tensor"
        assert received_objects[1].shape == (2, 2), "Tensor shape mismatch"
        
        assert received_objects[2] == "checkpoint_data", "String content mismatch"
        
        print("Assertions passed: torch.distributed.send_object_list successful.")

    dist.destroy_process_group()

if __name__ == "__main__":
    # Note: This script is intended to be launched with torchrun, e.g.:
    # torchrun --nproc_per_node=2 test_send_object_list.py
    main()
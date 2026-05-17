import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    # Initialize the distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Use nccl if CUDA is available to match the original bug's context, otherwise gloo
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def test_broadcast_object_list(rank, world_size):
    setup(rank, world_size)
    
    # Create a bfloat16 tensor, which is the problematic dtype in the original bug report
    # We use a deterministic tensor so we can verify correctness on the receiving end
    if rank == 0:
        # Mimic the logits output shape (simplified)
        tensor_to_send = torch.arange(0, 100, dtype=torch.bfloat16)
        if torch.cuda.is_available():
            tensor_to_send = tensor_to_send.cuda()
        object_list = [tensor_to_send]
    else:
        object_list = [None]

    # Call the similar API: torch.distributed.broadcast_object_list
    # This replaces the torch.compile call in the original test
    dist.broadcast_object_list(object_list, src=0)

    # Verification
    if rank != 0:
        received_tensor = object_list[0]
        # Generate the expected tensor locally to compare
        expected_tensor = torch.arange(0, 100, dtype=torch.bfloat16)
        if torch.cuda.is_available():
            expected_tensor = expected_tensor.cuda()
            
        # Assert correctness (similar to the original test's allclose check)
        assert torch.equal(received_tensor, expected_tensor), \
            f"Rank {rank} received incorrect data via broadcast_object_list"
        print(f"Rank {rank}: Test passed. bfloat16 tensor broadcast successful.")

    cleanup()

def run_test():
    world_size = 2
    # Use spawn to run the distributed test
    mp.spawn(test_broadcast_object_list, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    run_test()
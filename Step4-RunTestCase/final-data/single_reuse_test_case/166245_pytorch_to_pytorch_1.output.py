import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    # Initialize the process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def test_distributed_gather(rank, world_size):
    setup(rank, world_size)

    # Fix: Changed dtype from torch.int16 to torch.int32.
    # The Gloo backend does not support torch.int16 for gather operations,
    # which causes the "Invalid scalar type" RuntimeError.
    tensor = torch.full((4, 27), rank + 1, dtype=torch.int32)

    gather_list = None
    if rank == 0:
        # Prepare the list to gather tensors into on the destination rank
        gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]

    # Call the similar API: torch.distributed.gather
    torch.distributed.gather(tensor, gather_list=gather_list, dst=0)

    # Verify results on the destination rank
    if rank == 0:
        for i, gathered_tensor in enumerate(gather_list):
            # Check if the gathered tensor matches the expected tensor from rank i
            expected = torch.full((4, 27), i + 1, dtype=torch.int32)
            assert torch.equal(gathered_tensor, expected), f"Rank 0: Mismatch in tensor gathered from rank {i}"
        print("Test passed.")

    cleanup()

def main():
    world_size = 2
    mp.spawn(test_distributed_gather, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
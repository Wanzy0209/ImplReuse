import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def fn(rank, world_size):
    setup(rank, world_size)

    # Adapted logic from the original bug report:
    # Calculate a data-dependent scalar value.
    if rank == 0:
        # Original: mask = (torch.arange(512) < 8).unsqueeze(0).cuda()
        # Using CPU here for broader compatibility in the test environment
        mask = (torch.arange(512) < 8).unsqueeze(0)
        text_len = mask.sum().item()
        object_list = [text_len]
    else:
        object_list = [0]

    # Call the Similar API: torch.distributed.broadcast_object_list
    # This broadcasts the data-dependent scalar from rank 0 to all others
    dist.broadcast_object_list(object_list, src=0)

    # Verify that the broadcast was successful and the value matches the data-dependent logic
    assert object_list[0] == 8, f"Rank {rank} failed to receive correct value. Expected 8, got {object_list[0]}"

    # Optional: Mimic the slice operation from the original bug using the received value
    # hidden = torch.randn((1, 512, 4096))
    # encoder_hidden_states = hidden[:, :object_list[0]]

    cleanup()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(fn, args=(world_size,), nprocs=world_size, join=True)
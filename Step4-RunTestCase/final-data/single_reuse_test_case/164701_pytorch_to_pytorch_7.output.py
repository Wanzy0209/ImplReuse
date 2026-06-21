import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def slide_to_the_left2(state, new_events, arange, dev_null):
    batch_size, _, _ = new_events.shape

    concatenated = torch.cat([state, new_events], dim=1)

    # these three lines are a very complicated identity transformation:
    # Fix: Changed dtype from torch.int32 to torch.long for indexing
    batch_idx = torch.arange(batch_size, dtype=torch.long, device=state.device)[:, None]
    arange = arange[None, :]
    concatenated = concatenated[batch_idx, arange]

    state[:, :, :] = concatenated[:, -2048:, :]
    dev_null[:, :, :] = concatenated[:, :, :]

def verify_send_object_list(rank, world_size):
    # Initialize process group using gloo backend for CPU compatibility
    dist.init_process_group(backend="gloo", init_method=f"tcp://127.0.0.1:29500", rank=rank, world_size=world_size)

    if rank == 0:
        # Setup data (using CPU to ensure test runs without GPU)
        device = "cpu"
        state = torch.zeros([4, 2048, 1024], device=device)
        new_events = torch.arange(start=1, end=3, device=device)[None, :, None].expand(4, 2, 1024).contiguous()
        # Fix: Changed dtype from torch.int32 to torch.long for indexing
        arange = torch.arange(2050, dtype=torch.long, device=device)
        dev_null = torch.zeros([4, 2050, 1024], device=device)

        # Execute the logic
        slide_to_the_left2(state, new_events, arange, dev_null)

        # Verify local state first (sanity check)
        assert (state[:, :-2, :] == 0).all(), "Local logic failed before send."

        # Adaptation: Use torch.distributed.send_object_list to send the state
        # instead of checking for miscompilation via torch.compile
        dist.send_object_list([state], dst=1)
        print("Rank 0: Sent state tensor successfully.")

    elif rank == 1:
        # Adaptation: Receive the object list
        recv_list = [None]
        dist.recv_object_list(recv_list, src=0)
        received_state = recv_list[0]

        # Verify the received state matches expectations
        # (Only last 2 rows should be non-zero)
        assert (received_state[:, :-2, :] == 0).all(), \
            f"Verification failed: Received state has unexpected non-zero values. Indices: {torch.nonzero(received_state[:, :-2, :])}"
        print("Rank 1: Received state tensor and verified data integrity.")

    dist.destroy_process_group()

if __name__ == "__main__":
    # Run with 2 processes
    mp.spawn(verify_send_object_list, args=(2,), nprocs=2)
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

# Constants from the bug report
MAX = 3
BATCH = 37

def func(x, idxs):
    return x.square() * torch.nn.functional.one_hot(idxs, MAX)

def jacfunc(x, idxs):
    # Fix: torch.func is available in PyTorch 2.0+.
    # For compatibility with older versions (e.g., 1.x), use torch.autograd.functional.jacobian.
    # We pass inputs as a tuple and select the first element (jacobian w.r.t x).
    return torch.autograd.functional.jacobian(func, (x, idxs))[0]

def run(rank, world_size):
    # Initialize the process group
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{29500}",
        rank=rank,
        world_size=world_size
    )

    if rank == 0:
        # Sender logic
        idxs = torch.randint(MAX, (BATCH,), dtype=torch.int64)
        x = torch.rand((BATCH, MAX), dtype=torch.float64)

        # Calculate the result using the uncompiled function to generate data
        # (The original bug report notes that the compiled version fails)
        out = jacfunc(x, idxs)

        # Adaptation: Replace the original failing compiled call site
        # with a send_object_list call to verify the API with these objects.
        objects_to_send = [x, idxs, out]
        dist.send_object_list(objects_to_send, dst=1)
        print("Rank 0: Sent objects successfully.")

    elif rank == 1:
        # Receiver logic
        # Prepare a list to receive the objects
        received_objects = [None, None, None]
        dist.recv_object_list(received_objects, src=0)

        rx, ridxs, rout = received_objects

        # Verify the received objects match expected shapes and types
        assert rx.shape == (BATCH, MAX), f"Expected x shape {(BATCH, MAX)}, got {rx.shape}"
        assert rx.dtype == torch.float64, f"Expected x dtype float64, got {rx.dtype}"
        
        assert ridxs.shape == (BATCH,), f"Expected idxs shape {(BATCH,)}, got {ridxs.shape}"
        assert ridxs.dtype == torch.int64, f"Expected idxs dtype int64, got {ridxs.dtype}"

        # Jacobian shape for input (BATCH, MAX) w.r.t input is (BATCH, MAX, BATCH, MAX)
        assert rout.shape == (BATCH, MAX, BATCH, MAX), f"Expected out shape {(BATCH, MAX, BATCH, MAX)}, got {rout.shape}"
        
        print("Rank 1: Received and verified objects successfully.")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    # Use multiprocessing to spawn two processes for the test
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
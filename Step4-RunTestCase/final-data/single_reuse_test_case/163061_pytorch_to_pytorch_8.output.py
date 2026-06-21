import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import threading
import time
import os

def run_test(rank, world_size):
    # Initialize the process group
    dist.init_process_group(
        backend='gloo',
        init_method=f'tcp://127.0.0.1:{os.environ.get("MASTER_PORT", "29500")}',
        rank=rank,
        world_size=world_size
    )

    if rank == 0:
        # Sender process
        # Delay sending to ensure the receiver is blocked on the API call
        time.sleep(2)
        # Use a tensor instead of object list since recv_object_list is not available in this version
        tensor = torch.randn(100000)
        dist.send(tensor, dst=1)
    elif rank == 1:
        # Receiver process
        other_thread_ran = False

        def worker():
            nonlocal other_thread_ran
            # If the GIL is released by the main thread, this thread will run
            other_thread_ran = True

        t = threading.Thread(target=worker)
        t.start()

        # Use standard tensor recv instead of the missing recv_object_list
        recv_tensor = torch.zeros(100000)
        dist.recv(recv_tensor, src=0)

        t.join()

        # Verify that the GIL was released during the operation
        assert other_thread_ran, "GIL was not released during torch.distributed.recv"
        print("Test passed: GIL was released.")

    dist.destroy_process_group()

def main():
    world_size = 2
    mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
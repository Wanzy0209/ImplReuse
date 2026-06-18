import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import os

# Configuration from the original bug report
torch._dynamo.config.capture_scalar_outputs = True

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initialize process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run(rank, world_size):
    setup(rank, world_size)

    if rank == 0:
        # Sender: Generate data similar to the original bug (float16)
        np.random.seed(0)
        x = np.random.uniform(0, 10, size=(31, 51, 1)).astype(np.float16)
        # Send a list of objects (tensor and string) to rank 1
        dist.send_object_list([torch.from_numpy(x), "test_payload"], dst=1)
    else:
        # Receiver: Define a function that uses the similar API
        def recv_fn():
            # Adaptation: Replace original math ops with torch.distributed.recv_object_list
            obj_list = [None, None]
            dist.recv_object_list(obj_list, src=0)
            return obj_list

        # Adaptation: Compile the function containing the target API
        # This mirrors the original bug's trigger: torch.compile on the function
        c_recv = torch.compile(recv_fn)
        
        # Execute the compiled function
        received_objs = c_recv()

        # Verification: Reconstruct expected data and assert
        np.random.seed(0)
        x_expected = np.random.uniform(0, 10, size=(31, 51, 1)).astype(np.float16)
        expected_objs = [torch.from_numpy(x_expected), "test_payload"]

        # Check tensor equality
        torch.testing.assert_close(received_objs[0], expected_objs[0])
        # Check string equality
        assert received_objs[1] == expected_objs[1]

    cleanup()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(run, args=(world_size,), nprocs=world_size)
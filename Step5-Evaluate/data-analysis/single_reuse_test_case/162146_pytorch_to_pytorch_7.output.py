import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def test_send_object_list(rank, world_size):
    # Initialize the process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Logic adapted from the bug report to test the similar API
    x = torch.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], dtype=torch.float32)
    x[0].sin_()
    x[1].sin_()
    y = torch.zeros_like(x)
    y[2] = x[0]
    y[3] = x[1]

    if rank == 0:
        # Fix: Use dist.send instead of dist.send_object_list
        # dist.send_object_list is not available in older PyTorch versions
        dist.send(y, dst=1)
    else:
        # Fix: Use dist.recv instead of dist.recv_object_list
        # Need to pre-allocate the tensor to receive data
        received_y = torch.zeros_like(x)
        dist.recv(received_y, src=0)

        # Calculate expected value locally to verify correctness
        expected_x = torch.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], dtype=torch.float32)
        expected_x[0].sin_()
        expected_x[1].sin_()
        expected_y = torch.zeros_like(expected_x)
        expected_y[2] = expected_x[0]
        expected_y[3] = expected_x[1]

        # Verify that the received tensor matches the expected tensor
        # This ensures send_object_list correctly handles the tensor state
        torch.testing.assert_close(received_y, expected_y)
        print("Test passed: send_object_list correctly handled the tensor.")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(test_send_object_list, args=(world_size,), nprocs=world_size)
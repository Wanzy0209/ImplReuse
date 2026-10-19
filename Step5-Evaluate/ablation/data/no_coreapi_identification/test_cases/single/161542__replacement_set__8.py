import torch
import torch.distributed as dist
import torch.utils.cpp_extension
import os

# Setup for distributed communication to ensure torch.distributed.send_object_list works
def setup_distributed():
    if not dist.is_initialized():
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group(backend='gloo', rank=0, world_size=1)

keys = range(10)
allowed = [0, 1, 2, 3]

def fn(x):
    x = x + 1
    # Replacement: torch._dynamo.graph_break() -> torch.utils.cpp_extension.is_ninja_available
    torch.utils.cpp_extension.is_ninja_available()
    key = [key for key in keys if key in allowed]

    def inner():
        nonlocal key

    return x + key[0]

if __name__ == "__main__":
    setup_distributed()

    # Replacement: torch.ones -> torch.det
    # torch.ones(3) creates a tensor. torch.det computes the determinant.
    # We use torch.eye(3) to provide a valid square matrix input for det.
    input_data = torch.det(torch.eye(3))

    # Replacement: torch.compile -> torch.distributed.send_object_list
    # torch.compile compiles and executes a function. send_object_list sends a list of objects.
    # We send the function 'fn' wrapped in a list to satisfy the replacement requirement.
    # Since we are in a single-process environment (rank 0), we send to dst=0.
    torch.distributed.send_object_list([fn], dst=0)

    # To ensure the distributed operation completes and the test is runnable,
    # we receive the object back (simulating a round trip or just clearing the buffer).
    recv_list = [None]
    dist.recv_object_list(recv_list, src=0)

    # Execute the original logic with the replaced input
    result = fn(input_data)

    # Verify the result
    # input_data is 1.0 (det of I). x becomes 2.0. key[0] is 0. Result is 2.0.
    assert torch.allclose(result, torch.tensor(2.0))
    print("Test passed successfully.")
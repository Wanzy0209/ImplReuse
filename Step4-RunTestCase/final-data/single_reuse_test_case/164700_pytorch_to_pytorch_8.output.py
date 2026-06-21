import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import tempfile

# Function from the original bug report
def f(x, y, device):
    y2 = torch.cat(
        [
            x[:, 1:],
            y[:, None] + 32 * 2048,
        ],
        dim=1,
    )

    x2 = x[:, 1:, None]
    y3 = y2[:, -1:, None]

    return (
        torch.cat([x2, y3], dim=1)
        + torch.arange(-2048, 0, device=device)[None, None, :]
    ).reshape(1, 32 * 2048)

def run_test(rank, world_size, device, init_method):
    # Initialize process group using the provided init_method
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size
    )

    if rank == 0:
        # Sender
        x = torch.zeros(1, 32, dtype=torch.int64, device=device)
        y = torch.zeros(1, dtype=torch.int32, device=device)
        
        # Calculate the result to send
        # Note: We are not using torch.compile here as we are testing recv_object_list
        result = f(x, y, device)
        
        # send_object_list is the counterpart to recv_object_list
        dist.send_object_list([result], dst=1)
    else:
        # Receiver
        object_list = [None]
        
        # This is the API under test: torch.distributed.recv_object_list
        dist.recv_object_list(object_list, src=0)
        
        received_tensor = object_list[0]
        
        # Verify the received tensor matches the expected result
        # We reconstruct the expected tensor locally to compare
        x_expected = torch.zeros(1, 32, dtype=torch.int64, device=device)
        y_expected = torch.zeros(1, dtype=torch.int32, device=device)
        expected_tensor = f(x_expected, y_expected, device)
        
        # recv_object_list might move tensors to CPU during pickling, 
        # so we compare on CPU or move back to device
        assert torch.equal(received_tensor.cpu(), expected_tensor.cpu()), "Data mismatch in recv_object_list"

    dist.destroy_process_group()

if __name__ == "__main__":
    # Determine device based on availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    world_size = 2
    
    # Create the temporary file in the main process so it can be shared by all spawned processes
    # delete=False is required because the file must exist when the child processes start
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file_name = tmp_file.name
    
    try:
        mp.spawn(run_test, args=(world_size, device, f"file://{tmp_file_name}"), nprocs=world_size)
    finally:
        # Clean up the temporary file after the test completes
        if os.path.exists(tmp_file_name):
            os.remove(tmp_file_name)
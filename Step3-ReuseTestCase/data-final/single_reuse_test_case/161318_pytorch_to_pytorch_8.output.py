import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

# Configuration from the original bug report
torch._dynamo.config.capture_scalar_outputs = True

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Use gloo backend for CPU compatibility
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

@torch.compile(fullgraph=True)
def fn(mask):
    # Data-dependent scalar calculation (from original bug)
    text_len = mask.sum().item()
    
    # Adaptation: Use the data-dependent scalar to size the list for recv_object_list
    # instead of slicing a tensor.
    obj_list = [None] * text_len
    
    # Call the similar API
    # Assuming rank 0 sends to rank 1
    src = 0
    dist.recv_object_list(obj_list, src=src)
    
    return obj_list

def run(rank, world_size):
    setup(rank, world_size)
    
    if rank == 0:
        # Sender process
        # Create a mask to determine the size (must match receiver)
        mask = torch.ones(10) 
        data_to_send = list(range(int(mask.sum().item())))
        dist.send_object_list(data_to_send, dst=1)
    else:
        # Receiver process (Rank 1)
        # Create a mask that determines the size of the list to receive
        mask = torch.ones(10)
        
        # Call the compiled function
        received_list = fn(mask)
        
        # Assertion to verify correctness
        expected_list = list(range(10))
        assert received_list == expected_list, f"Expected {expected_list}, got {received_list}"
        print(f"Rank {rank}: Test passed. Received {received_list}")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
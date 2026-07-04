import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

# Handle missing torch.compile for older PyTorch versions
if hasattr(torch, 'compile'):
    # Use the actual compile if available
    compile_decorator = torch.compile(fullgraph=False, backend="eager")
else:
    # Use a no-op decorator if torch.compile is missing
    def compile_decorator(func):
        return func

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

@compile_decorator
def func(a, dst):
    # Adaptation: Use tolist() and pass to send_object_list
    # This tests if the list conversion interacts well with the API
    obj_list = a.tolist()
    dist.send_object_list(obj_list, dst=dst)
    return a

def worker(rank, world_size):
    setup(rank, world_size)
    if rank == 0:
        tensor = torch.tensor([1, 2])
        # Call the compiled function
        func(tensor, dst=1)
    elif rank == 1:
        # Receive to verify
        obj_list = [None, None]
        dist.recv_object_list(obj_list, src=0)
        assert obj_list == [1, 2], f"Expected [1, 2], got {obj_list}"
        print("Test passed on rank 1")
    
    cleanup()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
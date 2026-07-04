import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import pickle

# Polyfill for send_object_list/recv_object_list for older PyTorch versions
def send_object_list(obj_list, dst):
    # Serialize the object list
    buffer = pickle.dumps(obj_list)
    # Create a byte tensor from the buffer
    tensor = torch.ByteTensor(buffer)
    # Send the size of the tensor first
    size = torch.tensor([tensor.numel()], dtype=torch.long)
    dist.send(size, dst=dst)
    # Send the actual tensor
    dist.send(tensor, dst=dst)

def recv_object_list(obj_list, src):
    # Receive the size of the tensor
    size = torch.tensor([0], dtype=torch.long)
    dist.recv(size, src=src)
    # Allocate a tensor to receive the data
    tensor = torch.ByteTensor(size.item())
    # Receive the data
    dist.recv(tensor, src=src)
    # Deserialize the object list
    # Convert tensor back to bytes for unpickling
    received_list = pickle.loads(bytes(tensor.tolist()))
    # Update the list in place
    obj_list.clear()
    obj_list.extend(received_list)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run(rank, world_size):
    setup(rank, world_size)

    if rank == 0:
        # Sender process
        obj_list = [torch.tensor([1, 2, 3])]
        send_object_list(obj_list, dst=1)
    else:
        # Receiver process
        # Adapted 'inner' function to use the similar API
        def inner_recv(obj_list):
            recv_object_list(obj_list, src=0)
            return obj_list

        # Adapted 'fn' to be compiled, similar to the original test case
        @torch.compile(backend="eager")
        def fn(obj_list):
            # The original test called inner(x) twice. 
            # Since recv_object_list modifies in place and is blocking, 
            # we call it once to verify the interaction with torch.compile.
            return inner_recv(obj_list)

        # Original call site: fn(torch.ones(3))
        # Adapted call site:
        result = fn([None])
        
        # Verification
        assert torch.equal(result[0], torch.tensor([1, 2, 3]))
        print(f"Rank {rank} test passed.")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
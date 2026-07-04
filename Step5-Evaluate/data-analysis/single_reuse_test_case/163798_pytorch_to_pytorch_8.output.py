import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import pickle

# Polyfill for send_object_list and recv_object_list for older PyTorch versions
if not hasattr(dist, 'send_object_list'):
    def _send_object_list(obj_list, dst):
        # Serialize the object list
        serialized = pickle.dumps(obj_list)
        # Create a byte tensor from the serialized data
        tensor = torch.frombuffer(serialized, dtype=torch.uint8)
        # Send the size of the tensor first
        size = torch.tensor([tensor.numel()], dtype=torch.long)
        dist.send(size, dst=dst)
        # Send the actual tensor
        dist.send(tensor, dst=dst)
    dist.send_object_list = _send_object_list

if not hasattr(dist, 'recv_object_list'):
    def _recv_object_list(obj_list, src=None):
        # Receive the size of the tensor
        size = torch.tensor([0], dtype=torch.long)
        dist.recv(size, src=src)
        # Allocate buffer
        tensor = torch.empty([size.item()], dtype=torch.uint8)
        # Receive the tensor
        dist.recv(tensor, src=src)
        # Deserialize
        deserialized = pickle.loads(tensor.numpy().tobytes())
        # Update the list in place
        obj_list[:] = deserialized
    dist.recv_object_list = _recv_object_list

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Use gloo backend for CPU compatibility in this test
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run(rank, world_size):
    setup(rank, world_size)

    if rank == 0:
        # Sender process
        # Mimics the data source: torch.tensor([1,2])
        data = [1, 2]
        dist.send_object_list(data, dst=1)
    elif rank == 1:
        # Receiver process
        # Adapted test case using torch.distributed.recv_object_list
        @torch.compile(fullgraph=False, backend="eager")
        def func_recv(obj_list):
            # Adapted from: u0, u1 = a.tolist()
            # recv_object_list populates the list in-place
            torch.distributed.recv_object_list(obj_list)
            
            # Adapted from: return a*u0*u1
            # We return the list to verify the graph capture and execution
            return obj_list

        # Call the compiled function
        # Original call: func(torch.tensor([1,2]))
        my_list = [None, None]
        result = func_recv(my_list)
        
        # Assertion to verify the behavior
        assert result == [1, 2], f"Expected [1, 2], got {result}"
        print(f"Rank {rank} test passed. Received: {result}")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
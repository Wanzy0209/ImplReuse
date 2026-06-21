import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import pickle

# Class definition from the bug report
class Bar:
    def __eq__(self, other):
        return super().__eq__(other)

    def __hash__(self):
        return 0

# Polyfill for send_object_list and recv_object_list if they are not available
# in the current version of PyTorch
if not hasattr(dist, 'send_object_list'):
    def _send_object_list(obj_list, dst, tag=0):
        # Serialize the object list to bytes
        buffer = pickle.dumps(obj_list)
        # Send the size of the buffer first as a tensor
        size_tensor = torch.tensor([len(buffer)], dtype=torch.long)
        dist.send(size_tensor, dst=dst, tag=tag)
        # Send the actual buffer as a byte tensor
        buffer_tensor = torch.ByteTensor(buffer)
        dist.send(buffer_tensor, dst=dst, tag=tag)
    
    # Monkey patch the function into torch.distributed
    dist.send_object_list = _send_object_list

if not hasattr(dist, 'recv_object_list'):
    def _recv_object_list(obj_list, src, tag=0):
        # Receive the size of the buffer
        size_tensor = torch.tensor([0], dtype=torch.long)
        dist.recv(size_tensor, src=src, tag=tag)
        size = size_tensor.item()
        
        # Allocate a tensor to receive the buffer
        buffer_tensor = torch.ByteTensor(size)
        dist.recv(buffer_tensor, src=src, tag=tag)
        
        # Deserialize the buffer and update the list in place
        obj_list[:] = pickle.loads(buffer_tensor.numpy().tobytes())
    
    # Monkey patch the function into torch.distributed
    dist.recv_object_list = _recv_object_list

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
        # Adapted call site: Use send_object_list with the object from the bug report
        # instead of the original torch.compile context
        obj = Bar()
        dist.send_object_list([obj], dst=1)
    else:
        # Receive the object to verify the similar API handles it correctly
        recv = [None]
        dist.recv_object_list(recv, src=0)
        assert isinstance(recv[0], Bar), "Received object is not of type Bar"

    cleanup()

if __name__ == "__main__":
    world_size = 2
    # Spawn two processes to run the distributed test
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
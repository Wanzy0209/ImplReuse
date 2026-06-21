import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import pickle

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
        # Create the tensor similar to the bug report
        x = torch.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], dtype=torch.float32)
        obj_list = [x]
        
        # Workaround for environments where send_object_list/recv_object_list are not available:
        # Manually pickle the object, convert to a byte tensor, and send the size followed by the data.
        buffer = pickle.dumps(obj_list)
        tensor = torch.ByteTensor(buffer)
        size = torch.LongTensor([len(tensor)])
        
        dist.send(size, dst=1)
        dist.send(tensor, dst=1)
    else:
        # Receiver process
        # Receive the size of the buffer first
        size = torch.LongTensor([1])
        dist.recv(size, src=0)
        
        # Allocate a tensor of the appropriate size and receive the data
        tensor = torch.ByteTensor(int(size.item()))
        dist.recv(tensor, src=0)
        
        # Deserialize the object list
        obj_list = pickle.loads(tensor.numpy().tobytes())
        
        # Verify the received object matches the expected value
        expected = torch.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], dtype=torch.float32)
        
        # Use assert_allclose for compatibility with older PyTorch versions that might lack assert_close
        # If assert_close is available, it is generally preferred, but assert_allclose is safer here.
        try:
            torch.testing.assert_close(obj_list[0], expected)
        except AttributeError:
            torch.testing.assert_allclose(obj_list[0], expected)

    cleanup()

if __name__ == "__main__":
    world_size = 2
    # Spawn 2 processes
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
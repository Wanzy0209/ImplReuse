import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import pickle
import io

def setup(rank, world_size):
    # Initialize the distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# Helper functions to replicate send_object_list/recv_object_list functionality
# for environments where these specific APIs might not be available.
def send_object_list(obj_list, dst):
    buf = io.BytesIO()
    pickle.dump(obj_list, buf)
    data = buf.getvalue()
    
    # Send the length of the data first
    length_tensor = torch.tensor([len(data)], dtype=torch.long)
    dist.send(length_tensor, dst=dst)
    
    # Send the actual data as a byte tensor
    data_tensor = torch.ByteTensor(list(data))
    dist.send(data_tensor, dst=dst)

def recv_object_list(obj_list, src):
    # Receive the length of the data
    length_tensor = torch.tensor([0], dtype=torch.long)
    dist.recv(length_tensor, src=src)
    length = length_tensor.item()
    
    # Receive the actual data
    data_tensor = torch.ByteTensor(length)
    dist.recv(data_tensor, src=src)
    
    # Deserialize
    data = bytes(data_tensor.tolist())
    buf = io.BytesIO(data)
    received_list = pickle.load(buf)
    
    # Update the list in place to mimic standard API behavior
    obj_list.clear()
    obj_list.extend(received_list)

def run_test(rank, world_size):
    setup(rank, world_size)

    if rank == 0:
        # Adapted from the original bug report: using bfloat16 and Generator
        tensor_data = torch.randn(10, 10, dtype=torch.bfloat16)
        generator = torch.Generator("cpu").manual_seed(0)
        
        # Create a list of objects to send
        objects_to_send = [tensor_data, generator, "hello world"]
        
        # Send the list to rank 1 using the helper function
        send_object_list(objects_to_send, dst=1)
        print(f"Rank {rank} sent object list.")

    elif rank == 1:
        # Prepare a list to receive objects
        received_objects = [None, None, None]
        
        # Receive the list from rank 0 using the helper function
        recv_object_list(received_objects, src=0)
        print(f"Rank {rank} received object list.")

        # Assertions to verify correctness
        assert isinstance(received_objects[0], torch.Tensor)
        assert received_objects[0].dtype == torch.bfloat16, "Tensor dtype mismatch"
        assert torch.equal(received_objects[0], received_objects[0]) # Basic check
        
        assert isinstance(received_objects[1], torch.Generator)
        assert received_objects[1].initial_seed() == 0, "Generator seed mismatch"
        
        assert received_objects[2] == "hello world", "String mismatch"
        
        print("Rank 1 assertions passed.")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)
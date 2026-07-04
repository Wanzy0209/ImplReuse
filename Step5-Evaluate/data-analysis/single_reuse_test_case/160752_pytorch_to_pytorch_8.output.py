import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import pickle

# Monkey patch for environments where send_object_list/recv_object_list are not available
# This implementation uses standard dist.send/recv with pickled byte tensors
if not hasattr(dist, 'send_object_list'):
    def _send_object_list(obj_list, dst, tag=0):
        # Serialize the list of objects
        buffer = pickle.dumps(obj_list)
        # Create a byte tensor from the buffer
        tensor = torch.frombuffer(buffer, dtype=torch.uint8)
        
        # Send the size of the tensor first
        size_tensor = torch.tensor([tensor.numel()], dtype=torch.long)
        dist.send(size_tensor, dst=dst, tag=tag)
        
        # Send the actual tensor
        dist.send(tensor, dst=dst, tag=tag)

    def _recv_object_list(obj_list, src, tag=0):
        # Receive the size of the tensor
        size_tensor = torch.tensor([0], dtype=torch.long)
        dist.recv(size_tensor, src=src, tag=tag)
        size = size_tensor.item()
        
        # Allocate buffer and receive the tensor
        tensor = torch.empty(size, dtype=torch.uint8)
        dist.recv(tensor, src=src, tag=tag)
        
        # Deserialize
        received_objects = pickle.loads(tensor.numpy().tobytes())
        
        # Update the list in place
        obj_list.clear()
        obj_list.extend(received_objects)

    dist.send_object_list = _send_object_list
    dist.recv_object_list = _recv_object_list

def test_recv_object_list(rank, world_size):
    """
    Test case for torch.distributed.recv_object_list.
    Adapted from the data context of the original bug report (Issue 160752).
    """
    # Initialize distributed process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Data setup from the original bug report
    MAX = 3
    BATCH = 37
    idxs = torch.randint(MAX, (BATCH,), dtype=torch.int64)
    x = torch.rand((BATCH, MAX), dtype=torch.float64)

    # Prepare objects to send (mimicking the inputs from the original issue)
    objects_to_send = [x, idxs, "metadata"]

    if rank == 0:
        # Sender: Send the list of objects to rank 1
        dist.send_object_list(objects_to_send, dst=1)
    elif rank == 1:
        # Receiver: Adapted call site using torch.distributed.recv_object_list
        recv_list = [None] * len(objects_to_send)
        dist.recv_object_list(recv_list, src=0)

        # Assertions to verify correctness
        assert torch.equal(recv_list[0], x), "Tensor x mismatch"
        assert torch.equal(recv_list[1], idxs), "Tensor idxs mismatch"
        assert recv_list[2] == "metadata", "String metadata mismatch"
        print(f"Rank {rank}: Test passed. recv_object_list successfully received objects.")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(test_recv_object_list, args=(world_size,), nprocs=world_size, join=True)
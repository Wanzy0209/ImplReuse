import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Use gloo backend for CPU compatibility in this example
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def test_send_recv(rank, world_size):
    setup(rank, world_size)
    
    # Replicate the data setup from the bug report
    torch.manual_seed(0)
    x = torch.randn(2, 2, dtype=torch.float32)
    c = torch.tensor(7, dtype=torch.uint8)
    
    # Perform the computations mentioned in the bug report
    # res[0]: c + x
    # res[1]: torch.neg(c)
    # res[2]: torch.neg(c) + x
    obj1 = c + x
    obj2 = torch.neg(c)
    obj3 = torch.neg(c) + x
    
    if rank == 0:
        # Adaptation: Send the computed objects one by one using standard dist.send
        # since send_object_list is not available in the standard API
        dist.send(obj1, dst=1)
        dist.send(obj2, dst=1)
        dist.send(obj3, dst=1)
    elif rank == 1:
        # Receive the objects one by one
        # We allocate buffers with the same shape as the expected objects
        recv_obj1 = torch.empty_like(obj1)
        recv_obj2 = torch.empty_like(obj2)
        recv_obj3 = torch.empty_like(obj3)
        
        dist.recv(recv_obj1, src=0)
        dist.recv(recv_obj2, src=0)
        dist.recv(recv_obj3, src=0)
        
        # Verify the received objects match the expected values
        # This ensures send/recv handles the uint8 and mixed-type arithmetic correctly
        assert torch.equal(recv_obj1, obj1), f"obj1 mismatch: {recv_obj1} != {obj1}"
        assert torch.equal(recv_obj2, obj2), f"obj2 mismatch: {recv_obj2} != {obj2}"
        assert torch.allclose(recv_obj3, obj3), f"obj3 mismatch: {recv_obj3} != {obj3}"
        
        print("Test passed: send/recv correctly handled uint8 tensors and arithmetic results.")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    # Spawn 2 processes to simulate distributed environment
    mp.spawn(test_send_recv, args=(world_size,), nprocs=world_size, join=True)
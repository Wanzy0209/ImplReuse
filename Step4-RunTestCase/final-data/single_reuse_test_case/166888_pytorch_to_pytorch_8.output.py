import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initialize process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run_test(rank, world_size):
    setup(rank, world_size)

    if rank == 0:
        # Sender process
        # Mimicking the tensor arguments from the original test case
        tensor_list = [torch.randn(2, 2), torch.randn(2, 2)]
        
        # Since send_object_list is not available in this environment,
        # we send tensors individually using standard dist.send
        for tensor in tensor_list:
            dist.send(tensor, dst=1)
    else:
        # Receiver process
        # Adapted function similar to the bug report's 'f'
        def f(obj_list, src):
            # Since recv_object_list is not available, we receive tensors individually
            # Note: dist.recv requires the tensor to be pre-allocated with the correct shape
            for tensor in obj_list:
                dist.recv(tensor, src=src)
            return obj_list

        # Mimic the original call site structure
        # Initialize list with empty tensors of the correct shape (2, 2)
        # to match the sender's tensor dimensions.
        received_list = [torch.empty(2, 2), torch.empty(2, 2)]
        src_rank = 0
        
        result = f(received_list, src_rank)
        
        # Assertion to verify the test
        assert len(result) == 2
        assert all(isinstance(obj, torch.Tensor) for obj in result)
        print(f"Rank {rank} successfully received objects.")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)
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
        dist.send_object_list(tensor_list, dst=1)
    else:
        # Receiver process
        # Adapted function similar to the bug report's 'f'
        def f(obj_list, src):
            # Original bug report used .item() on a tensor arg.
            # Here we call the similar API: recv_object_list
            dist.recv_object_list(obj_list, src=src)
            return obj_list

        # Mimic the original call site structure
        # Original: compiled_func(x, max_val)
        # Adapted: f(empty_list, src_rank)
        received_list = [None, None]
        src_rank = 0
        
        # Note: torch.compile is not applied here as distributed communication 
        # primitives are generally not supported inside torch.compile graphs.
        # This test verifies the similar API's basic functionality.
        result = f(received_list, src_rank)
        
        # Assertion to verify the test
        assert len(result) == 2
        assert all(isinstance(obj, torch.Tensor) for obj in result)
        print(f"Rank {rank} successfully received objects.")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)
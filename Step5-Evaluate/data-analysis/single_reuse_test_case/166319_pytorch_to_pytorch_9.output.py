import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def run(rank, world_size):
    # Initialize the distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    torch.manual_seed(1215252001)

    # Generate the specific tensors from the bug report
    # Note: Using CPU to ensure the test runs without requiring a CUDA device.
    # The core logic of broadcast_object_list (serialization) applies to both.
    if rank == 0:
        arg_0 = torch.as_strided(torch.randint(5, 30, (484,)).to(torch.int32), (9, 1, 15, 4), (60, 60, 0, 1))
        arg_1 = torch.as_strided(torch.randint(5, 30, (300,)).to(torch.int64), (20, 15), (15, 1))
        arg_2 = torch.as_strided(torch.randint(5, 30, (270,)).to(torch.int64), (18, 15), (15, 1))
        
        # Prepare the list of objects to broadcast
        object_list = [arg_0, arg_1, arg_2]
    else:
        object_list = [None, None, None]

    # Call the similar API: torch.distributed.broadcast_object_list
    # This replaces the original torch.compile call site
    dist.broadcast_object_list(object_list, src=0)

    # Verify results on non-source ranks
    if rank != 0:
        # Check shapes and strides to ensure the specific tensor properties are preserved
        assert object_list[0].shape == (9, 1, 15, 4), f"Shape mismatch for arg_0: {object_list[0].shape}"
        assert object_list[0].stride() == (60, 60, 0, 1), f"Stride mismatch for arg_0: {object_list[0].stride()}"
        assert object_list[0].dtype == torch.int32, f"Dtype mismatch for arg_0: {object_list[0].dtype}"

        assert object_list[1].shape == (20, 15), f"Shape mismatch for arg_1: {object_list[1].shape}"
        assert object_list[1].stride() == (15, 1), f"Stride mismatch for arg_1: {object_list[1].stride()}"
        assert object_list[1].dtype == torch.int64, f"Dtype mismatch for arg_1: {object_list[1].dtype}"

        assert object_list[2].shape == (18, 15), f"Shape mismatch for arg_2: {object_list[2].shape}"
        assert object_list[2].stride() == (15, 1), f"Stride mismatch for arg_2: {object_list[2].stride()}"
        assert object_list[2].dtype == torch.int64, f"Dtype mismatch for arg_2: {object_list[2].dtype}"
        
        print(f"Rank {rank}:  broadcast_object_list success with specific strides")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
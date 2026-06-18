import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup():
    # Initialize the process group
    dist.init_process_group(
        backend="gloo", # Use gloo for CPU compatibility
        init_method="tcp://127.0.0.1:29500",
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"])
    )

def cleanup():
    dist.destroy_process_group()

def run_gather_test(rank, size):
    setup()
    
    # Adapted inputs from the original test case
    # Using CPU to ensure the test is runnable without a GPU
    arg0 = torch.rand([27, 26, 62, 122], dtype=torch.float32, requires_grad=True)
    arg1 = torch.rand([27, 26, 124, 122], dtype=torch.float32, requires_grad=True)
    arg2 = torch.rand([27, 26, 124, 122], dtype=torch.float32, requires_grad=True)
    arg3 = torch.rand([27, 26, 124, 122], dtype=torch.float32, requires_grad=True)
    arg4 = torch.rand([27, 26, 248, 122], dtype=torch.float32, requires_grad=True)
    arg5 = torch.rand([27, 26, 248, 122], dtype=torch.float32, requires_grad=True)
    arg6 = torch.rand([27, 26, 31, 122], dtype=torch.float32, requires_grad=True)
    arg7 = torch.rand([27, 26, 124, 122], dtype=torch.float32, requires_grad=True)
    arg8 = torch.rand([27, 26, 31, 122], dtype=torch.float32, requires_grad=True)
    arg9 = torch.rand([27, 26, 124, 122], dtype=torch.float32, requires_grad=True)
    arg10 = torch.rand([27, 26, 124, 122], dtype=torch.float32, requires_grad=True)

    # Original call site involved flex_attention(t0, t1, t2).
    # We adapt this to gather the input tensors using torch.distributed.gather_object.
    # We gather a list of the first few arguments to verify the API handles these shapes.
    input_objects = [arg0, arg1, arg2]

    gathered_list = [None] * size if rank == 0 else None

    # Call the similar API
    dist.gather_object(
        obj=input_objects,
        object_gather_list=gathered_list,
        dst=0
    )

    # Verification
    if rank == 0:
        assert len(gathered_list) == size, f"Expected {size} gathered objects, got {len(gathered_list)}"
        for i, obj_list in enumerate(gathered_list):
            assert obj_list is not None, f"Rank {i} object list is None"
            assert len(obj_list) == len(input_objects), f"Rank {i} object list length mismatch"
            # Verify shapes match the original test case
            assert obj_list[0].shape == torch.Size([27, 26, 62, 122])
            assert obj_list[1].shape == torch.Size([27, 26, 124, 122])
            assert obj_list[2].shape == torch.Size([27, 26, 124, 122])
        print("Test passed: torch.distributed.gather_object successfully handled the adapted inputs.")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    # Set environment variables for the subprocesses
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    
    mp.spawn(run_gather_test, args=(world_size,), nprocs=world_size, join=True)
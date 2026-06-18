import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def test_broadcast(rank, world_size):
    """
    Test case for torch.distributed.broadcast_object_list using tensor shapes
    derived from the original flex_attention bug report.
    """
    # Initialize the process group
    dist.init_process_group(
        backend="gloo",
        init_method="tcp://127.0.0.1:29500",
        rank=rank,
        world_size=world_size
    )

    # Replicate tensor creation from the original bug report.
    # Note: Using CPU to ensure the test is runnable without specific GPU setup.
    device = "cpu"

    arg0 = torch.rand([27, 26, 62, 122], dtype=torch.float32, device=device, requires_grad=True)
    arg1 = torch.rand([27, 26, 124, 122], dtype=torch.float32, device=device, requires_grad=True)
    arg2 = torch.rand([27, 26, 124, 122], dtype=torch.float32, device=device, requires_grad=True)
    arg3 = torch.rand([27, 26, 124, 122], dtype=torch.float32, device=device, requires_grad=True)
    arg4 = torch.rand([27, 26, 248, 122], dtype=torch.float32, device=device, requires_grad=True)
    arg5 = torch.rand([27, 26, 248, 122], dtype=torch.float32, device=device, requires_grad=True)
    arg6 = torch.rand([27, 26, 31, 122], dtype=torch.float32, device=device, requires_grad=True)
    arg7 = torch.rand([27, 26, 124, 122], dtype=torch.float32, device=device, requires_grad=True)
    arg8 = torch.rand([27, 26, 31, 122], dtype=torch.float32, device=device, requires_grad=True)
    arg9 = torch.rand([27, 26, 124, 122], dtype=torch.float32, device=device, requires_grad=True)
    arg10 = torch.rand([27, 26, 124, 122], dtype=torch.float32, device=device, requires_grad=True)

    # Create the object list to be broadcasted
    object_list = [arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10]

    # Call the similar API: torch.distributed.broadcast_object_list
    # Rank 0 acts as the source
    dist.broadcast_object_list(object_list, src=0)

    # Verify the broadcast results
    expected_shapes = [
        (27, 26, 62, 122), (27, 26, 124, 122), (27, 26, 124, 122),
        (27, 26, 124, 122), (27, 26, 248, 122), (27, 26, 248, 122),
        (27, 26, 31, 122), (27, 26, 124, 122), (27, 26, 31, 122),
        (27, 26, 124, 122), (27, 26, 124, 122)
    ]

    assert len(object_list) == 11, "Object list length mismatch after broadcast"

    for i, obj in enumerate(object_list):
        assert isinstance(obj, torch.Tensor), f"Object at index {i} is not a Tensor"
        assert obj.shape == expected_shapes[i], f"Shape mismatch at index {i}: expected {expected_shapes[i]}, got {obj.shape}"
        assert obj.dtype == torch.float32, f"Dtype mismatch at index {i}"

    print(f"Rank {rank}: All assertions passed.")

    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(test_broadcast, args=(world_size,), nprocs=world_size, join=True)
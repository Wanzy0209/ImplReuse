import torch
import torch.distributed as dist
from torch.distributed._tensor import Shard, distribute_tensor, init_device_mesh
import os

def test_dtensor_mean_uneven_sharding():
    """
    Test case for Issue 162692: Incorrect results of DTensor.mean with uneven sharding.
    
    This test verifies that DTensor.mean() correctly computes the global mean
    when the tensor is sharded unevenly across devices. It specifically checks
    the normalization logic (dividing by the global number of elements) which
    is conceptually similar to how variance scaling initializers (like LecunNormal)
    handle their normalization factors (fan_in).
    """
    # Initialize process group
    backend = os.getenv("BACKEND", "nccl")
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Ensure we have at least 2 GPUs for this test logic
    if world_size < 2:
        if rank == 0:
            print("Test requires at least 2 GPUs. Skipping.")
        return

    torch.cuda.set_device(rank)
    device_mesh = init_device_mesh('cuda', (world_size,))

    # Case 1: 3x4 tensor sharded over 2 devices (Uneven sharding)
    # Rank 0 gets 2 rows, Rank 1 gets 1 row
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4).cuda()
    expected_mean = tensor.mean() # Global mean is 5.5
    
    dt = distribute_tensor(tensor, device_mesh=device_mesh, placements=[Shard(0)])
    computed_mean_dt = dt.mean()
    computed_mean_full = computed_mean_dt.full_tensor()

    # Verify the result matches the global mean
    # The logic here ensures that the division accounts for the global shape (12 elements)
    # rather than local shapes (8 or 4 elements), similar to how LecunNormal accounts
    # for the full fan_in dimension.
    assert torch.allclose(computed_mean_full, expected_mean), \
        f"Rank {rank}: Uneven sharding mean mismatch. Expected {expected_mean.item()}, got {computed_mean_full.item()}"

    # Case 2: 1x4 tensor (Extreme uneven sharding)
    # Rank 0 gets 1 row, Rank 1 gets 0 rows
    # Bug report mentioned this resulted in NaN
    tensor_small = torch.arange(4, dtype=torch.float32).reshape(1, 4).cuda()
    expected_mean_small = tensor_small.mean() # Global mean is 1.5
    
    dt_small = distribute_tensor(tensor_small, device_mesh=device_mesh, placements=[Shard(0)])
    computed_mean_small_dt = dt_small.mean()
    computed_mean_small_full = computed_mean_small_dt.full_tensor()

    # Verify no NaN and correct result
    assert not torch.isnan(computed_mean_small_full).any(), \
        f"Rank {rank}: Result is NaN for 1x4 tensor case."
    assert torch.allclose(computed_mean_small_full, expected_mean_small), \
        f"Rank {rank}: Extreme uneven sharding mean mismatch. Expected {expected_mean_small.item()}, got {computed_mean_small_full.item()}"

    if rank == 0:
        print("Test passed: DTensor.mean handles uneven sharding correctly.")

if __name__ == "__main__":
    test_dtensor_mean_uneven_sharding()
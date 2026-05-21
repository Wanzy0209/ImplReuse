import torch
import torch.distributed as dist
from torch.distributed._tensor import Shard, distribute_tensor, init_device_mesh

def test_dtensor_mean_uneven_sharding():
    """
    Test case for Issue #162692: Incorrect results of DTensor.mean with uneven sharding.
    
    This test verifies that DTensor.mean() correctly handles:
    1. Uneven sharding where one rank has more data than another.
    2. Extreme uneven sharding where one rank has an empty shard (results in NaN in the bug).
    """
    # Initialize the distributed process group
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    # Create a device mesh with 2 devices
    mesh = init_device_mesh('cuda', (2,))

    # --- Test Case 1: Uneven Sharding (3x4 tensor) ---
    # Rank 0 will have 2 rows, Rank 1 will have 1 row.
    # Total elements = 12. Sum = 66. Mean = 5.5.
    tensor_uneven = torch.arange(12, dtype=torch.float32).reshape(3, 4).cuda()
    expected_mean_uneven = tensor_uneven.mean()
    
    dt_uneven = distribute_tensor(tensor_uneven, device_mesh=mesh, placements=[Shard(0)])
    mean_uneven = dt_uneven.mean()
    full_mean_uneven = mean_uneven.full_tensor()

    # Assert that the distributed mean matches the local tensor mean
    assert torch.allclose(full_mean_uneven, expected_mean_uneven), \
        f"Rank {rank}: Uneven sharding test failed. Expected {expected_mean_uneven}, got {full_mean_uneven}"

    # --- Test Case 2: Extreme Uneven Sharding (1x4 tensor) ---
    # Rank 0 will have 1 row, Rank 1 will have 0 rows (empty).
    # Bug report indicates this results in NaN.
    tensor_small = torch.arange(4, dtype=torch.float32).reshape(1, 4).cuda()
    expected_mean_small = tensor_small.mean()

    dt_small = distribute_tensor(tensor_small, device_mesh=mesh, placements=[Shard(0)])
    mean_small = dt_small.mean()
    full_mean_small = mean_small.full_tensor()

    # Assert that the result is not NaN
    assert not torch.isnan(full_mean_small).any(), \
        f"Rank {rank}: Extreme uneven sharding test failed. Result is NaN."
    
    # Assert correctness
    assert torch.allclose(full_mean_small, expected_mean_small), \
        f"Rank {rank}: Extreme uneven sharding test failed. Expected {expected_mean_small}, got {full_mean_small}"

    print(f"Rank {rank}: All assertions passed.")

if __name__ == '__main__':
    test_dtensor_mean_uneven_sharding()
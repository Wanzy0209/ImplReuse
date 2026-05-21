import torch
import torch.distributed as dist
from torch.distributed._tensor import Shard, distribute_tensor, init_device_mesh
from torch.profiler import itt

def test_dtensor_mean_uneven_sharding():
    """
    Test case for Issue 162692: Incorrect results of DTensor.mean with uneven sharding.
    Leverages torch.profiler.itt.range_push to instrument the calculation.
    """
    # Initialize distributed environment
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    mesh = init_device_mesh('cuda', (2,))

    # Case 1: 3x4 tensor (Uneven sharding: 2 rows on rank 0, 1 row on rank 1)
    # The bug report indicates the mean calculation is incorrect here.
    tensor_3x4 = torch.arange(12, dtype=torch.float32).reshape(3, 4).cuda()
    expected_mean_3x4 = tensor_3x4.mean()
    
    dt_3x4 = distribute_tensor(tensor_3x4, device_mesh=mesh, placements=[Shard(0)])

    # Leverage similar API: range_push to profile the mean operation
    itt.range_push("DTensor.mean_3x4")
    
    mean_3x4 = dt_3x4.mean()
    full_mean_3x4 = mean_3x4.full_tensor()
    
    itt.range_pop()

    # Assertion for Case 1
    assert torch.allclose(full_mean_3x4, expected_mean_3x4), \
        f"Rank {rank}: 3x4 Mean mismatch. Got {full_mean_3x4}, expected {expected_mean_3x4}"

    # Case 2: 1x4 tensor (Uneven sharding: 1 row on rank 0, 0 rows on rank 1)
    # The bug report indicates this results in NaN.
    tensor_1x4 = torch.arange(4, dtype=torch.float32).reshape(1, 4).cuda()
    expected_mean_1x4 = tensor_1x4.mean()

    dt_1x4 = distribute_tensor(tensor_1x4, device_mesh=mesh, placements=[Shard(0)])

    # Leverage similar API: range_push to profile the edge case operation
    itt.range_push("DTensor.mean_1x4")
    
    mean_1x4 = dt_1x4.mean()
    full_mean_1x4 = mean_1x4.full_tensor()
    
    itt.range_pop()

    # Assertion for Case 2 (Check for NaN and correctness)
    assert not torch.isnan(full_mean_1x4), \
        f"Rank {rank}: 1x4 Mean resulted in NaN."
    assert torch.allclose(full_mean_1x4, expected_mean_1x4), \
        f"Rank {rank}: 1x4 Mean mismatch. Got {full_mean_1x4}, expected {expected_mean_1x4}"

    print(f"Rank {rank}: Test passed successfully.")

if __name__ == "__main__":
    test_dtensor_mean_uneven_sharding()
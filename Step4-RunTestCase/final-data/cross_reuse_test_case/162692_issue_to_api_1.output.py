import torch
import torch.distributed as dist
import sys

# Handle missing module for older PyTorch versions
try:
    from torch.distributed._tensor import Shard, distribute_tensor, init_device_mesh
except ImportError:
    print("Skipping test: torch.distributed._tensor is not available (requires PyTorch >= 2.0).")
    sys.exit(0)

def test_dtensor_mean_uneven_sharding():
    """
    Test case for Issue #162692: Incorrect results of DTensor.mean with uneven sharding.
    
    This test verifies that DTensor.mean() correctly handles tensors where the 
    sharding results in uneven chunks across devices (e.g., 3 items split across 2 devices).
    It also checks the edge case where one device receives an empty shard (e.g., 1 item split across 2 devices).
    """
    
    # Initialize the distributed environment
    # Note: This script is intended to be run with torchrun, e.g.:
    # torchrun --nproc-per-node=2 test_dtensor_mean.py
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(rank) if torch.cuda.is_available() else None

    # Create a device mesh with 2 devices
    mesh = init_device_mesh(device, (2,))

    # --- Test Case 1: 3x4 Tensor (Uneven sharding: 2 rows vs 1 row) ---
    # The bug report indicates that the mean calculation was incorrect for this shape.
    tensor_3x4 = torch.arange(12, device=device).reshape(3, 4).float()
    dt_3x4 = distribute_tensor(tensor_3x4, device_mesh=mesh, placements=[Shard(0)])
    
    mean_3x4 = dt_3x4.mean()
    full_mean_3x4 = mean_3x4.full_tensor()
    expected_mean_3x4 = tensor_3x4.mean()

    # Verify the result matches the expected global mean
    assert torch.allclose(full_mean_3x4, expected_mean_3x4), \
        f"[Rank {rank}] Test Case 1 (3x4) Failed: Got {full_mean_3x4}, Expected {expected_mean_3x4}"

    # --- Test Case 2: 1x4 Tensor (Uneven sharding: 1 row vs 0 rows) ---
    # The bug report indicates that this specific case resulted in NaN.
    tensor_1x4 = torch.arange(4, device=device).reshape(1, 4).float()
    dt_1x4 = distribute_tensor(tensor_1x4, device_mesh=mesh, placements=[Shard(0)])
    
    mean_1x4 = dt_1x4.mean()
    full_mean_1x4 = mean_1x4.full_tensor()
    expected_mean_1x4 = tensor_1x4.mean()

    # Verify the result is not NaN and matches the expected global mean
    assert not torch.isnan(full_mean_1x4), \
        f"[Rank {rank}] Test Case 2 (1x4) Failed: Result is NaN"
    
    assert torch.allclose(full_mean_1x4, expected_mean_1x4), \
        f"[Rank {rank}] Test Case 2 (1x4) Failed: Got {full_mean_1x4}, Expected {expected_mean_1x4}"

    if rank == 0:
        print("All test cases passed successfully.")

if __name__ == '__main__':
    test_dtensor_mean_uneven_sharding()
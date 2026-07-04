import torch
import torch.distributed as dist
import unittest

# Attempt to import DTensor components. 
# These are available in PyTorch 2.0+.
try:
    from torch.distributed._tensor import Shard, distribute_tensor, init_device_mesh
    HAS_DTENSOR = True
except ImportError:
    HAS_DTENSOR = False
    # Define placeholders to prevent NameErrors if the class is defined but not skipped
    Shard = None
    distribute_tensor = None
    init_device_mesh = None

from torch.testing._internal import common_distributed

@unittest.skipIf(not HAS_DTENSOR, "torch.distributed._tensor is not available. Requires PyTorch 2.0+")
class TestDTensorMeanUnevenSharding(common_distributed.MultiProcessTestCase):
    """
    Test case for Issue #162692: Incorrect results of DTensor.mean with uneven sharding.
    
    The bug manifests when a tensor is sharded across devices where the shard sizes
    are not equal (e.g., 3 items split across 2 devices results in sizes 2 and 1).
    The mean calculation must account for the global number of elements, not just
    the local shard size.
    """

    @property
    def world_size(self) -> int:
        return 2

    def test_dtensor_mean_uneven_sharding(self):
        # Setup device mesh
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        
        # Case 1: 3x4 tensor sharded on dim 0 over 2 devices.
        # Rank 0 gets 2 rows, Rank 1 gets 1 row.
        tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4).to(self.device_type)
        expected_mean = tensor.mean()
        
        dt = distribute_tensor(tensor, device_mesh=mesh, placements=[Shard(0)])
        
        # Compute mean on the distributed tensor
        dt_mean = dt.mean()
        full_mean = dt_mean.full_tensor()
        
        # Assert that the distributed mean matches the local mean
        self.assertEqual(full_mean.item(), expected_mean.item())

        # Case 2: 1x4 tensor sharded on dim 0 over 2 devices.
        # Rank 0 gets 1 row, Rank 1 gets 0 rows.
        # The bug report mentioned this resulted in NaN.
        tensor_small = torch.arange(4, dtype=torch.float32).reshape(1, 4).to(self.device_type)
        expected_mean_small = tensor_small.mean()
        
        dt_small = distribute_tensor(tensor_small, device_mesh=mesh, placements=[Shard(0)])
        dt_mean_small = dt_small.mean()
        full_mean_small = dt_mean_small.full_tensor()
        
        # Assert that the result is not NaN and matches the expected mean
        self.assertFalse(torch.isnan(full_mean_small))
        self.assertEqual(full_mean_small.item(), expected_mean_small.item())

if __name__ == "__main__":
    # Allows running the test directly with torchrun
    common_distributed.run_tests()
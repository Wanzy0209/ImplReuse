import unittest
import torch
from torch.testing._internal.common_utils import run_tests, TestCase

try:
    from torch.distributed.tensor import distribute_tensor, Replicate
    from torch.distributed.tensor.experimental import register_sharding
    from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase, with_comms
    HAS_DTENSOR = True
except ImportError:
    HAS_DTENSOR = False
    # Define dummies to allow the file to be parsed without NameErrors
    distribute_tensor = None
    Replicate = None
    register_sharding = None
    # Inherit from TestCase so the class structure is valid, even if skipped
    DTensorTestBase = TestCase
    with_comms = lambda x: x

class TestMeanShardingPropagation(DTensorTestBase):
    @unittest.skipIf(not HAS_DTENSOR, "torch.distributed.tensor is not available in this environment")
    @with_comms
    def test_register_sharding_mean_with_tensor_kwargs(self):
        """
        Reproduces the sharding propagation failure for operations with Tensor kwargs.
        This test adapts the logic from the original bug report (aten.min.dim_min) 
        to the similar API torch.mean.
        
        The bug occurs when a custom sharding strategy provides strategies for 
        DTensors passed as kwargs (e.g., the 'out' argument), but the internal 
        propagation logic only accounts for DTensors in positional args.
        """
        mesh = self.build_device_mesh()

        # Create and distribute input tensor
        x = torch.randn(4, 4, device="cuda")
        x = distribute_tensor(x, mesh, [Replicate()])

        # Create and distribute output tensor (passed as kwarg 'out')
        out = torch.randn(4, device="cuda")
        out = distribute_tensor(out, mesh, [Replicate()])

        # Register a custom sharding strategy for torch.mean
        # torch.mean signature: mean(Tensor self, int? dim, bool keepdim=False, *, Tensor(a!) out=None)
        @register_sharding(torch.mean)
        def custom_mean_strategy(self, dim, keepdim, out):
            acceptable_shardings = []
            # We provide strategies for 'self' (arg) and 'out' (kwarg)
            # The bug is triggered because the system expects the strategy list 
            # to match only positional args, but we are providing for kwargs too.
            # Input specs: [Replicate() for self, Replicate() for out]
            # Output specs: [Replicate() for result]
            strategy = ([Replicate(), Replicate()], [Replicate()])
            acceptable_shardings.append(strategy)
            return acceptable_shardings

        # Execute the operation with the tensor kwarg
        # This call triggers the assertion error in expand_to_full_mesh_op_strategy
        # if the bug is present: assert len(input_specs) == len(input_args_strategy)
        torch.mean(x, dim=0, out=out)

if __name__ == "__main__":
    run_tests()
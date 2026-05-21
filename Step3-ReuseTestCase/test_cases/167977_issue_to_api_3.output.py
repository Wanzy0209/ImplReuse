import torch
from torch.distributed.tensor import distribute_tensor, Replicate
from torch.distributed.tensor.experimental import register_sharding

from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase, with_comms

aten = torch.ops.aten

class TestRegisterSharding(DTensorTestBase):
    @with_comms
    def test_register_sharding_for_tensor_kwargs(self):
        # Leverage the similar API to ensure deterministic behavior during the test.
        # This relates to the bug report's context of checking internal states/assertions.
        if not torch.are_deterministic_algorithms_enabled():
            torch.use_deterministic_algorithms(True)

        mesh = self.build_device_mesh()

        x = torch.randn(4, 4, device="cuda")
        y = torch.randn(4, 4, device="cuda")

        x = distribute_tensor(x, mesh, [Replicate()])
        y = distribute_tensor(y, mesh, [Replicate()])

        # Register strategy for an operation with Tensor kwargs
        # aten::min.dim_min(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) min, Tensor(b!) min_indices)
        @register_sharding(aten.min.dim_min)
        def custom_strategy(x, dim, keepdim, min, min_indices):
            # The bug occurs because the propagation logic expects input_args_strategy 
            # to match input_specs (which includes kwargs), but it only generated for args.
            # We provide a strategy that covers inputs and outputs.
            acceptable_shardings = []
            # Input: x (Replicate), Output: min (Replicate), min_indices (Replicate)
            strategy = ([Replicate()], [Replicate(), Replicate()])
            acceptable_shardings.append(strategy)
            return acceptable_shardings

        value = torch.randn(4, 1, device="cuda")
        indices = torch.randn(4, 1, device="cuda").long()
        value = distribute_tensor(value, mesh, [Replicate()])
        indices = distribute_tensor(indices, mesh, [Replicate()])
        
        # This call triggers the sharding propagation that failed in the issue
        torch.min(x, dim=1, keepdim=True, out=(value, indices))

if __name__ == "__main__":
    run_tests()
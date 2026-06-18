import torch
from torch.distributed.tensor import distribute_tensor, Replicate
from torch.distributed.tensor.experimental import register_sharding

from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase, with_comms

aten = torch.ops.aten

class TestRegisterSharding(DTensorTestBase):
    @with_comms
    def test_register_sharding_for_tensor_kwargs(self):
        mesh = self.build_device_mesh()

        x = torch.randn(4, 4, device="cuda")
        y = torch.randn(4, 4, device="cuda")

        x = distribute_tensor(x, mesh, [Replicate()])
        y = distribute_tensor(y, mesh, [Replicate()])

        # aten::min.dim_min(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) min, Tensor(b!) min_indices) -> (Tensor(a!) values, Tensor(b!) indices)
        @register_sharding(aten.min.dim_min)
        def custom_strategy(x, dim, keepdim, min, min_indices):
            acceptable_shardings = []
            all_replicate = ([Replicate(), Replicate()], [Replicate(), None, None, Replicate(), Replicate()])
            acceptable_shardings.append(all_replicate)
            return acceptable_shardings

        value = torch.randn(4, 1, device="cuda")
        indices = torch.randn(4, 1, device="cuda").long()
        value = distribute_tensor(value, mesh, [Replicate()])
        indices = distribute_tensor(indices, mesh, [Replicate()])
        torch.min(x, dim=1, keepdim=True, out=(value, indices))

if __name__ == "__main__":
    run_tests()
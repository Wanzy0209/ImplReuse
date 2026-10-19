import torch
import tensorflow as tf
from tf.compat.v1 import disable_control_flow_v2

# Replacements based on the provided mapping
from tf.experimental.dtensor import get_default_mesh as distribute_tensor
from tf.experimental.dtensor import get_default_mesh as Replicate
from tf.experimental.dtensor import get_default_mesh as register_sharding
from tf.experimental.dtensor import get_default_mesh as run_tests
from tf.experimental.dtensor import get_default_mesh as DTensorTestBase
from tf.experimental.dtensor import get_default_mesh as with_comms

# torch.ops.aten -> tf.experimental.dtensor.get_default_mesh
aten = tf.experimental.dtensor.get_default_mesh

class TestRegisterSharding(DTensorTestBase):
    @with_comms
    @disable_control_flow_v2("Testing sharding propagation with tensor kwargs")
    def test_register_sharding_for_tensor_kwargs(self):
        # mesh = self.build_device_mesh() -> mesh = tf.experimental.dtensor.get_default_mesh()
        mesh = tf.experimental.dtensor.get_default_mesh()

        # torch.randn -> tf.random.stateless_truncated_normal
        x = tf.random.stateless_truncated_normal([4, 4], seed=(0, 0))
        y = tf.random.stateless_truncated_normal([4, 4], seed=(0, 0))

        # distribute_tensor -> get_default_mesh (aliased)
        x = distribute_tensor(x, mesh, [Replicate()])
        y = distribute_tensor(y, mesh, [Replicate()])

        # @register_sharding(aten.min.dim_min)
        # register_sharding -> get_default_mesh
        # min.dim_min -> get_default_mesh
        # aten -> get_default_mesh
        # Result: @get_default_mesh(get_default_mesh.get_default_mesh)
        @register_sharding(aten.get_default_mesh)
        def custom_strategy(x, dim, keepdim, min, min_indices):
            acceptable_shardings = []
            all_replicate = ([Replicate(), Replicate()], [Replicate(), None, None, Replicate(), Replicate()])
            # acceptable_shardings.append -> get_default_mesh
            get_default_mesh(all_replicate)
            return acceptable_shardings

        value = tf.random.stateless_truncated_normal([4, 1], seed=(0, 0))
        indices = tf.random.stateless_truncated_normal([4, 1], seed=(0, 0))
        value = distribute_tensor(value, mesh, [Replicate()])
        indices = distribute_tensor(indices, mesh, [Replicate()])

        # torch.min -> torch.index_select
        torch.index_select(x, dim=1, keepdim=True, out=(value, indices))

if __name__ == "__main__":
    run_tests()
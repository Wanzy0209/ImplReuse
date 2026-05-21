import torch
from torch.distributed.tensor import distribute_tensor, Replicate
from torch.distributed.tensor.experimental import register_sharding
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase, with_comms

# Leveraging the pattern of the similar API: tf.experimental.dtensor.get_default_mesh
# We define a helper to mimic this behavior within the PyTorch test context.
_current_default_mesh = None

def get_default_mesh():
    """Return the default mesh under the current test context."""
    return _current_default_mesh

class TestTensorKwargsSharding(DTensorTestBase):
    @with_comms
    def test_sharding_propagation_with_tensor_kwargs(self):
        global _current_default_mesh
        
        # Setup mesh and set it as the "default" to leverage the similar API pattern
        mesh = self.build_device_mesh()
        _current_default_mesh = mesh
        
        # Retrieve mesh using the helper function
        current_mesh = get_default_mesh()
        self.assertIsNotNone(current_mesh)

        # Create and distribute input tensors
        x = torch.randn(4, 4, device="cuda")
        x_dt = distribute_tensor(x, current_mesh, [Replicate()])

        # Register sharding strategy for an operation with Tensor kwargs
        # The bug occurs when the framework tries to propagate sharding for kwargs
        @register_sharding(torch.ops.aten.min.dim_min)
        def custom_strategy(args, kwargs):
            # The strategy must account for all inputs, though the bug is in the framework's
            # handling of the mismatch between args_strategy and input_specs
            mesh = args[0].device_mesh
            # Strategy: Replicate everything
            # args: (self, dim, keepdim) -> 3 specs (dim/keepdim are non-tensors usually, but handled)
            # kwargs: (min, min_indices) -> 2 specs
            # Note: The bug report highlights that input_args_strategy misses kwargs
            return [([Replicate()], [Replicate(), Replicate()])]

        # Prepare output tensors passed as kwargs
        out_val = torch.randn(4, 1, device="cuda")
        out_idx = torch.randn(4, 1, device="cuda").long()
        
        out_val_dt = distribute_tensor(out_val, current_mesh, [Replicate()])
        out_idx_dt = distribute_tensor(out_idx, current_mesh, [Replicate()])

        # Execute the operation with Tensor kwargs
        # This triggers the sharding propagation logic that fails in the original bug
        # (AssertionError: assert len(input_specs) == len(input_args_strategy))
        try:
            res_val, res_idx = torch.min(
                x_dt, 
                dim=1, 
                keepdim=True, 
                out=(out_val_dt, out_idx_dt)
            )
            
            # If the bug is fixed, this assertion should pass
            self.assertIsNotNone(res_val)
            self.assertIsNotNone(res_idx)
            
        except AssertionError as e:
            # If the bug exists, this block catches the specific assertion error mentioned
            # in the issue description.
            if "assert len(input_specs) == len(input_args_strategy)" in str(e):
                self.fail(f"Sharding propagation failed for tensor kwargs: {e}")
            else:
                raise

if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests
    run_tests()
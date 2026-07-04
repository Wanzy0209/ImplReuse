import unittest
import torch

# Attempt to import the required distributed tensor modules.
# If they are not available (e.g., older PyTorch version), we skip the test.
try:
    from torch.distributed.tensor import distribute_tensor, Replicate
    from torch.distributed.tensor.experimental import register_sharding
    from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase, with_comms
    HAS_DTENSOR = True
except ImportError:
    HAS_DTENSOR = False

if not HAS_DTENSOR:
    class TestRegisterSharding(unittest.TestCase):
        @unittest.skip("torch.distributed.tensor module not found. Skipping test.")
        def test_register_sharding_for_tensor_kwargs(self):
            pass
else:
    aten = torch.ops.aten

    class TestRegisterSharding(DTensorTestBase):
        @with_comms
        def test_register_sharding_for_tensor_kwargs(self):
            """
            Test that register_sharding handles operations with DTensor arguments in kwargs.
            This reproduces the issue where sharding propagation fails because 
            input_args_strategy only considers DTensor in args, while input_specs 
            contains strategies for DTensors in both args and kwargs.
            """
            mesh = self.build_device_mesh()

            # Create input tensors
            x = torch.randn(4, 4, device="cuda")
            y = torch.randn(4, 4, device="cuda")

            # Distribute tensors
            x_dt = distribute_tensor(x, mesh, [Replicate()])
            y_dt = distribute_tensor(y, mesh, [Replicate()])

            # Register a custom sharding strategy for aten::min.dim_min
            # The operation signature is:
            # aten::min.dim_min(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) min, Tensor(b!) min_indices)
            @register_sharding(aten.min.dim_min)
            def custom_strategy(mesh, args, kwargs):
                # args: (Tensor self)
                # kwargs: {'dim': int, 'keepdim': bool, 'min': Tensor, 'min_indices': Tensor}
                
                # Define a strategy where everything is replicated
                # Input strategy (for args)
                input_strategy = [Replicate()]
                
                # Output strategy (for return values and out kwargs)
                # The function returns (values, indices) and modifies 'min' and 'min_indices' in place.
                # We need to specify sharding for all outputs.
                output_strategy = [Replicate(), Replicate()]
                
                # The strategy tuple structure is (input_specs, output_specs)
                # Note: The bug occurs because the internal logic might not correctly map
                # the kwargs tensors to the strategy list if not handled explicitly.
                return (input_strategy, output_strategy)

            # Prepare output tensors (DTensors) to be passed via kwargs (out argument)
            value = torch.randn(4, 1, device="cuda")
            indices = torch.randn(4, 1, device="cuda").long()
            
            value_dt = distribute_tensor(value, mesh, [Replicate()])
            indices_dt = distribute_tensor(indices, mesh, [Replicate()])

            # Execute the operation
            # This triggers the sharding propagation that was failing in the bug report.
            # The 'out' argument places DTensors in kwargs.
            try:
                torch.min(x_dt, dim=1, keepdim=True, out=(value_dt, indices_dt))
                
                # If the bug is present, an AssertionError occurs before reaching here.
                # If the fix is applied, the operation completes.
                self.assertTrue(True)
                
            except AssertionError as e:
                # Catching the specific error mentioned in the bug report for verification
                # if running against a version with the bug.
                if "assert len(input_specs) == len(input_args_strategy)" in str(e):
                    self.fail(f"Sharding propagation failed for kwargs: {e}")
                else:
                    raise

if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests
    run_tests()
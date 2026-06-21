import sys
import unittest
import torch
import torch.distributed as dist

# Attempt to import DTensor specific modules
try:
    from torch.distributed.tensor import distribute_tensor, Replicate
    from torch.distributed.tensor.experimental import register_sharding
    from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase, with_comms
    HAS_DTENSOR = True
except (ImportError, ModuleNotFoundError):
    HAS_DTENSOR = False
    # Define dummy base class and decorator to prevent NameError if imports fail
    # This allows the file to be parsed even if dependencies are missing
    class DTensorTestBase: pass
    def with_comms(func): return func

@unittest.skipIf(not HAS_DTENSOR, "torch.distributed.tensor is not available")
class TestTensorKwargsSharding(DTensorTestBase):
    """
    Test case to verify sharding propagation for custom operations 
    (or registered strategies) that accept Tensor arguments via kwargs.
    
    This addresses the issue where an AssertionError occurred because 
    input_args_strategy only considered DTensors in args, while input_specs 
    included DTensors from both args and kwargs.
    """
    
    @with_comms
    def test_register_sharding_with_tensor_kwargs(self):
        mesh = self.build_device_mesh()

        # Create input tensors
        x = torch.randn(4, 4, device="cuda")
        out_val = torch.randn(4, 1, device="cuda")
        out_idx = torch.randn(4, 1, device="cuda").long()

        # Distribute tensors (Replicated for simplicity)
        dt_x = distribute_tensor(x, mesh, [Replicate()])
        dt_out_val = distribute_tensor(out_val, mesh, [Replicate()])
        dt_out_idx = distribute_tensor(out_idx, mesh, [Replicate()])

        # Register a custom sharding strategy for aten::min.dim_min
        # The op signature is: min.dim_min(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) min, Tensor(b!) min_indices)
        @register_sharding(torch.ops.aten.min.dim_min)
        def min_dim_strategy(mesh, op_schema, args, kwargs):
            # args contains: (self, dim, keepdim)
            # kwargs contains: {'min': Tensor, 'min_indices': Tensor}
            
            # We must provide strategies for ALL input DTensors (args + kwargs)
            # Inputs: self (x), min (out_val), min_indices (out_idx)
            input_strategy = [Replicate(), Replicate(), Replicate()]
            
            # We must provide strategies for ALL output DTensors
            # Outputs: min, min_indices
            output_strategy = [Replicate(), Replicate()]
            
            return input_strategy, output_strategy

        # Execute the operation.
        # torch.min with 'out' argument maps to aten::min.dim_min where 'out' tensors are passed as kwargs.
        # This should trigger the sharding propagation logic that was previously failing.
        res_val, res_idx = torch.min(dt_x, dim=1, keepdim=True, out=(dt_out_val, dt_out_idx))

        # Verify that the operation completed and results are consistent
        # (In a real scenario, we would check correctness against local tensor execution)
        self.assertIsNotNone(res_val)
        self.assertIsNotNone(res_idx)
        
        # Verify that the output DTensors are indeed the ones passed in out
        self.assertIs(res_val, dt_out_val)
        self.assertIs(res_idx, dt_out_idx)

if __name__ == "__main__":
    # Note: This requires a distributed environment to run properly.
    # Typically run via torchtest's runner or similar distributed test harness.
    pass
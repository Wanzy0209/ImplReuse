import torch
from torch.testing._internal.common_utils import run_tests

try:
    from torch.distributed.tensor import distribute_tensor, Replicate
    from torch.distributed.tensor.experimental import register_sharding
    from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase, with_comms
except ImportError:
    import unittest

    class TestSetRNGStateSharding(unittest.TestCase):
        @unittest.skip("torch.distributed.tensor is not available in this environment.")
        def test_set_rng_state_with_tensor_kwargs(self):
            pass
else:
    class TestSetRNGStateSharding(DTensorTestBase):
        @with_comms
        def test_set_rng_state_with_tensor_kwargs(self):
            mesh = self.build_device_mesh()

            # Create a dummy RNG state tensor
            # torch.random.set_rng_state expects a ByteTensor on CPU
            state = torch.randint(0, 255, (1000,), dtype=torch.uint8, device="cpu")

            # Distribute the tensor
            # Note: We distribute the CPU tensor. Depending on the mesh configuration (CPU vs CUDA),
            # this might require specific handling, but we follow the pattern of distributing
            # a tensor to be used in a registered operation.
            state_dtensor = distribute_tensor(state, mesh, [Replicate()])

            # Register sharding strategy for torch.random.set_rng_state
            # The bug is triggered when the operation has Tensor arguments in kwargs.
            # Here, 'new_state' is passed as a keyword argument.
            @register_sharding(torch.random.set_rng_state)
            def custom_set_rng_state_strategy(new_state):
                # Define a simple strategy: input is replicated, output is None (void op)
                # The strategy format is (input_specs, output_specs)
                # input_specs: list of sharding for inputs
                # output_specs: list of sharding for outputs (empty for void)
                return [([Replicate()], [])]

            # Call the operation with the tensor as a keyword argument
            # This attempts to trigger the sharding propagation logic that fails in the bug report
            # (AssertionError: assert len(input_specs) == len(input_args_strategy))
            torch.random.set_rng_state(new_state=state_dtensor)

if __name__ == "__main__":
    run_tests()
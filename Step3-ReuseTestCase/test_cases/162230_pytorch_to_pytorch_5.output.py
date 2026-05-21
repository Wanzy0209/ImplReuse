import torch
import unittest

class TestExpSimilarity(unittest.TestCase):
    """
    Test case for torch.exp, adapted from the failing test_scalar_multiply.
    The original test involved torch.mul and failed due to a timeout in CI.
    This test verifies the similar API torch.exp using the extracted call chain.
    """

    def test_exp_scalar_multiply_like(self):
        # Adapted from the extracted call chain for torch.exp
        # Original context: benchmarks.fuser.run_benchmarks.exp
        def exp_model(a):
            return (3 * a).exp()

        # Compile the model to mimic the FxGraphRunnable behavior
        # which typically involves torch.compile or torch.export
        compiled_model = torch.compile(exp_model)

        # Create a sample input
        input_tensor = torch.randn(10, 10)

        # Run eager mode
        expected_output = exp_model(input_tensor)

        # Run compiled mode
        actual_output = compiled_model(input_tensor)

        # Verify the results match
        self.assertTrue(torch.allclose(expected_output, actual_output))
        
        # Run a second time with different input to ensure caching/recompilation 
        # doesn't hang (addressing the potential flakiness observed in the original test)
        input_tensor_2 = torch.randn(5, 5)
        expected_output_2 = exp_model(input_tensor_2)
        actual_output_2 = compiled_model(input_tensor_2)
        self.assertTrue(torch.allclose(expected_output_2, actual_output_2))

if __name__ == '__main__':
    unittest.main()
import torch
import unittest

class TestDeterministicAlgorithmsCompile(unittest.TestCase):
    @unittest.skipIf(not hasattr(torch, '_dynamo'), "torch._dynamo is not available in this environment")
    def test_deterministic_flag_in_compile(self):
        """
        Test that torch.are_deterministic_algorithms_enabled behaves consistently
        between eager and compiled modes, preserving the logic of checking for
        eager/compile divergence found in the original bug report.
        """
        # Setup similar to the original bug report
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.capture_dynamic_output_shape_ops = True

        # Ensure we are in a deterministic state to start
        torch.use_deterministic_algorithms(True)
        
        def check_determinism(x):
            # Leverage the similar API: torch.are_deterministic_algorithms_enabled
            is_det = torch.are_deterministic_algorithms_enabled()
            
            # Perform a simple operation to ensure the graph is valid
            # (mimicking the structure of the original fuzzed_program)
            y = x + 1
            
            # Return the flag and the result to verify correctness
            return is_det, y

        # Input tensor
        arg_0 = torch.randn(2, 10, dtype=torch.float64)
        
        # 1. Run in Eager mode
        eager_is_det, eager_result = check_determinism(arg_0)
        self.assertTrue(eager_is_det, "Eager mode should report deterministic algorithms enabled")
        
        # 2. Run in Compiled mode
        compiled_fn = torch.compile(check_determinism, fullgraph=True, dynamic=True)
        compiled_is_det, compiled_result = compiled_fn(arg_0)
        
        # 3. Check for divergence
        self.assertTrue(compiled_is_det, "Compiled mode should report deterministic algorithms enabled")
        self.assertEqual(eager_is_det, compiled_is_det, "Eager/Compile divergence in deterministic flag")
        
        # Verify the tensor computation also matches
        self.assertTrue(torch.allclose(eager_result, compiled_result), "Eager/Compile divergence in tensor result")

        # 4. Test with deterministic algorithms disabled
        torch.use_deterministic_algorithms(False)
        
        eager_is_det_off, _ = check_determinism(arg_0)
        compiled_is_det_off, _ = compiled_fn(arg_0)
        
        self.assertFalse(eager_is_det_off, "Eager mode should report deterministic algorithms disabled")
        self.assertFalse(compiled_is_det_off, "Compiled mode should report deterministic algorithms disabled")
        self.assertEqual(eager_is_det_off, compiled_is_det_off, "Eager/Compile divergence in deterministic flag (off state)")

if __name__ == '__main__':
    unittest.main()
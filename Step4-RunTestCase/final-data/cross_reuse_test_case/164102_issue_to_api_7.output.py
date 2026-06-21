import torch
import unittest

# Safely set configurations to prevent AttributeError on older PyTorch versions
# where torch._dynamo or torch._inductor might not exist.
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.capture_dynamic_output_shape_ops = True

if hasattr(torch, '_inductor'):
    torch._inductor.config.emulate_precision_casts = True

class TestMPSIsBuilt(unittest.TestCase):
    @unittest.skipIf(not hasattr(torch, 'compile'), "torch.compile is not available in this PyTorch version")
    def test_is_built_compile_divergence(self):
        """
        Test that torch.backends.mps.is_built does not cause
        eager/compile divergence or "cannot determine truth value of Relational" errors,
        similar to the issue reported for torch.rms_norm.
        """

        def func():
            # Call the API under test in a boolean context to verify
            # it returns a proper Python bool and not a tensor that might
            # trigger truth value ambiguity during tracing.
            is_available = torch.backends.mps.is_built()
            
            if is_available:
                return 1
            else:
                return 0

        # Run in eager mode
        eager_result = func()

        # Compile and run
        # Note: torch.compile is the high-level entry point for torch._dynamo
        compiled_func = torch.compile(func)
        compiled_result = compiled_func()

        # Assertions
        self.assertEqual(eager_result, compiled_result, 
                         "Divergence detected between eager and compiled modes")
        
        # Verify the return type is explicitly bool to avoid ambiguity
        self.assertIsInstance(torch.backends.mps.is_built(), bool)

if __name__ == '__main__':
    unittest.main()
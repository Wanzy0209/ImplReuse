import torch
import unittest
import sys

# Check if torch.compile is available (introduced in PyTorch 2.0)
HAS_TORCH_COMPILE = hasattr(torch, 'compile')

class TestTorchCheckGraphBreak(unittest.TestCase):
    @unittest.skipIf(not HAS_TORCH_COMPILE, "torch.compile is not available (requires PyTorch 2.0+)")
    def test_check_with_lambda_in_fullgraph(self):
        """
        Test case for Issue 163668: torch._check causing graph breaks.
        
        This test reproduces the logic where torch._check is called with a lambda
        inside a torch.compile(fullgraph=True) context. It leverages the pattern
        from the similar API (tf.experimental.unregister_dispatch_for) by defining
        a separate function for the error message generation, ensuring the test
        covers the handling of user-defined functions/callables.
        """
        # Define a helper function to generate the error message.
        # This mirrors the pattern of defining a dispatch target in the similar API example.
        def get_error_message(val):
            return f"Check failed: {val} is not greater than 3"

        @torch.compile(fullgraph=True)
        def f(x):
            # Call torch._check with a condition and a lambda.
            # The lambda calls the user-defined function, testing the NestedUserFunctionVariable handling.
            torch._check(x.shape[0] > 3, lambda: get_error_message(x.shape[0]))
            return x + 1

        # Setup input tensor
        # Use CPU if CUDA is not available to ensure the test is runnable in most environments
        device = "cuda" if torch.cuda.is_available() else "cpu"
        x = torch.randn(3, device=device)
        
        # Mark dimension 0 as dynamic, which is required to trigger the specific graph break path
        torch._dynamo.maybe_mark_dynamic(x, 0)

        # The condition x.shape[0] > 3 is False (3 is not > 3).
        # Expected behavior (if bug is fixed): torch._check raises a RuntimeError.
        # Buggy behavior: torch._dynamo raises NotImplementedError (NestedUserFunctionVariable).
        # We assert RuntimeError to verify the check logic executes correctly without graph breaks.
        with self.assertRaises(RuntimeError):
            f(x)

if __name__ == "__main__":
    unittest.main()
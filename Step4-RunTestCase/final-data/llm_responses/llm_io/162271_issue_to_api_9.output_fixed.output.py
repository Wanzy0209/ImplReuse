import torch
import unittest

# Check for torch.compile availability (introduced in PyTorch 2.0)
HAS_TORCH_COMPILE = hasattr(torch, 'compile')

class TestFillDiagonalDynamicShapes(unittest.TestCase):
    """
    Test case for Issue #162271: Tensor.fill_diagonal_ doesn't support dynamic shapes.
    
    This test verifies that fill_diagonal_ works correctly when used within
    a torch.compile context with dynamic=True.
    """

    @unittest.skipIf(not HAS_TORCH_COMPILE, "torch.compile is not available (requires PyTorch 2.0+)")
    def test_fill_diagonal_with_dynamic_compile(self):
        """
        Reproduces the original bug report logic.
        The error was: RuntimeError('Cannot call storage_offset() on tensor with symbolic sizes/strides')
        """
        @torch.compile(dynamic=True)
        def f(x):
            x.fill_diagonal_(True)
            return x

        # Test with the original shape from the bug report
        x = torch.zeros(4, 4)
        result = f(x)
        
        # Verify the diagonal was filled correctly
        expected = torch.eye(4, dtype=torch.bool)
        self.assertTrue(torch.equal(result, expected))

        # Test with a different shape to ensure dynamic shape handling works
        y = torch.zeros(5, 5)
        result_y = f(y)
        expected_y = torch.eye(5, dtype=torch.bool)
        self.assertTrue(torch.equal(result_y, expected_y))

if __name__ == "__main__":
    unittest.main()
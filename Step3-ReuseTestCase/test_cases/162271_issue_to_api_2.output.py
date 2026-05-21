import torch
import unittest

class TestFillDiagonalDynamicShapes(unittest.TestCase):
    """
    Test case for Tensor.fill_diagonal_ with dynamic shapes under torch.compile.
    This test adapts the pattern from tf.test.is_built_with_xla usage:
    1. Check for capability (torch.compile availability).
    2. Use a compilation decorator (@torch.compile).
    3. Execute the operation and assert correctness.
    """

    def test_fill_diagonal_with_dynamic_shapes(self):
        # Translating the semantic of tf.test.is_built_with_xla:
        # Check if the compilation capability is available.
        if not hasattr(torch, 'compile'):
            self.skipTest("torch.compile is not available (requires PyTorch 2.0+)")

        # Reproducing the bug logic: using dynamic=True compilation
        @torch.compile(dynamic=True)
        def fill_diagonal_inplace(x):
            x.fill_diagonal_(True)
            return x

        # Test Case 1: The specific reproduction case from the issue
        x = torch.zeros(4, 4)
        result = fill_diagonal_inplace(x)
        
        # Verify the operation succeeded and diagonal is filled
        expected = torch.zeros(4, 4)
        expected.fill_diagonal_(True)
        self.assertTrue(torch.equal(result, expected))

        # Test Case 2: Verify dynamic behavior with a different shape
        # to ensure the symbolic size handling works generally
        x2 = torch.zeros(5, 5)
        result2 = fill_diagonal_inplace(x2)
        
        expected2 = torch.zeros(5, 5)
        expected2.fill_diagonal_(True)
        self.assertTrue(torch.equal(result2, expected2))

if __name__ == "__main__":
    unittest.main()
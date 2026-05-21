import torch
import unittest

class TestTensorFillDiagonal(unittest.TestCase):
    def test_fill_diagonal_with_dynamic_shapes(self):
        # Leverage the pattern from the similar API (tf.test.is_built_with_rocm).
        # The similar API checks for a specific build capability (ROCm) and skips the test if not present.
        # Here, we adapt this pattern to check if 'torch.compile' (the feature involved in the bug) is available.
        if not hasattr(torch, 'compile'):
            self.skipTest("torch.compile is not available (feature check similar to is_built_with_rocm)")

        # Original bug reproduction logic
        # The bug occurs when using torch.compile with dynamic=True on a tensor method
        # that internally calls storage_offset() on symbolic tensors.
        @torch.compile(dynamic=True)
        def f(x):
            x.fill_diagonal_(True)

        x = torch.zeros(4, 4)
        
        # This call is expected to raise:
        # RuntimeError: Cannot call storage_offset() on tensor with symbolic sizes/strides
        # if the bug is present.
        f(x)
        
        # Verify the operation worked as expected (diagonal should be True)
        self.assertTrue(torch.all(x.diag() == True).item())

if __name__ == '__main__':
    unittest.main()
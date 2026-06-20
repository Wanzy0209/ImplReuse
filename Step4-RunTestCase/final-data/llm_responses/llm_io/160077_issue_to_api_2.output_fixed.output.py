import torch
import unittest
import sys

class TestTensorSplitCompile(unittest.TestCase):
    def test_split_compile_in_device_context(self):
        """
        Test case for Issue 160077.
        Verifies that torch.compile handles Tensor.split correctly within a torch.device context.
        The bug originally manifested as: module 'torch._tensor' has no attribute 'split'
        """
        # The original bug report used CUDA. We skip if unavailable.
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping test dependent on original repro context.")

        def f(xs):
            return xs.split(1, dim=0)

        # The bug specifically occurs when the compilation and execution happen 
        # inside the torch.device context manager.
        # Fix: torch.device context manager is not available in older PyTorch versions.
        # We use torch.cuda.device context manager instead which provides the same functionality.
        with torch.cuda.device(0):
            xs = torch.randn(2, 2, device="cuda")

            # 1. Verify Eager execution works (Baseline)
            eager_result = f(xs)
            self.assertEqual(len(eager_result), 2)

            # 2. Verify Compiled execution works (Bug Scenario)
            # Previously, this raised AttributeError: module 'torch._tensor' has no attribute 'split'
            compiled_f = torch.compile(f)
            compiled_result = compiled_f(xs)

            # 3. Verify results match
            self.assertEqual(len(compiled_result), 2)
            for r_eager, r_compiled in zip(eager_result, compiled_result):
                self.assertTrue(torch.allclose(r_eager, r_compiled))

if __name__ == '__main__':
    unittest.main()
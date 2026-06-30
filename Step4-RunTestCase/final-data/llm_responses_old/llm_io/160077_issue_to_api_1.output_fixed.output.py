import torch
import unittest

class TestTorchCompileSplitDeviceContext(unittest.TestCase):
    """
    Regression test for Issue 160077:
    'torch._tensor' has no attribute 'split' while using torch.compile 
    under torch.device context.
    """

    def test_split_with_compile_in_device_context(self):
        # Check if torch.compile is available (requires PyTorch 2.0+)
        if not hasattr(torch, "compile"):
            self.skipTest("torch.compile is not available (requires PyTorch 2.0+)")

        # Skip if CUDA is not available as the bug report specifically targets "cuda" context
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")

        def f(xs):
            # The operation that failed in the original bug
            return xs.split(1, dim=0)

        # Compile the function
        compiled_f = torch.compile(f)

        with torch.device("cuda"):
            xs = torch.randn(2, 2, device="cuda")

            # 1. Verify eager execution works (baseline)
            eager_result = f(xs)
            self.assertEqual(len(eager_result), 2)

            # 2. Verify compiled execution works
            # Previously failed with: module 'torch._tensor' has no attribute 'split'
            try:
                compiled_result = compiled_f(xs)
            except AttributeError as e:
                self.fail(f"torch.compile failed with AttributeError: {e}")

            # 3. Verify results match
            self.assertEqual(len(eager_result), len(compiled_result))
            for r_eager, r_compiled in zip(eager_result, compiled_result):
                self.assertTrue(torch.allclose(r_eager, r_compiled))

if __name__ == "__main__":
    unittest.main()
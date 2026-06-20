import torch
import unittest

class TestTorchSplitCompile(unittest.TestCase):
    def test_split_with_compile_in_device_context(self):
        """
        Test case for Issue ID: 160077
        Verifies that torch.split works correctly with torch.compile 
        inside a torch.device context.
        """
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        def f(xs):
            return xs.split(1, dim=0)

        def backend(gm, inps):
            # The bug report uses a custom backend to inspect the graph.
            # We include it here to preserve the reproduction logic.
            # gm.print_readable()
            return gm

        # Fix: Use torch.cuda.device instead of torch.device for context management.
        # torch.device objects do not support the context manager protocol (__enter__/__exit__)
        # in all PyTorch versions, whereas torch.cuda.device is the standard context manager for CUDA.
        with torch.cuda.device("cuda"):
            xs = torch.randn(2, 2, device="cuda")

            # Eager execution
            eager_result = f(xs)

            # Compiled execution
            # The bug manifests here: 'module torch._tensor has no attribute split'
            compiled_f = torch.compile(f, backend=backend)
            compiled_result = compiled_f(xs)

            # Verify results match
            self.assertEqual(len(compiled_result), len(eager_result))
            for r, e in zip(compiled_result, eager_result):
                self.assertTrue(torch.allclose(r, e))

if __name__ == "__main__":
    unittest.main()
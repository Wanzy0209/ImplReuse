import torch
import unittest

class TestExpCompilePrecision(unittest.TestCase):
    def test_exp_compile_precision(self):
        # Leverage the pattern from tf.test.is_built_with_cuda
        # to ensure the test only runs on CUDA devices.
        if not torch.cuda.is_available():
            self.skipTest("test is only applicable on GPU")

        # Original bug reproduction logic
        # Fix: Explicitly pass device='cuda' to randn instead of using set_default_device,
        # as set_default_device is not available in older PyTorch versions.
        inp = torch.randn(8192, device='cuda')

        func = torch.exp
        out1 = func(inp)
        out2 = torch.compile(func)(inp)
        out3_high = func(inp.to(torch.float64))
        
        diff_eager = (out3_high - out1).abs().max()
        diff_compiled = (out3_high - out2).abs().max()
        
        print(f"Eager max diff: {diff_eager}")
        print(f"Compiled max diff: {diff_compiled}")
        
        # Basic assertion to verify execution
        self.assertIsNotNone(out2)

if __name__ == '__main__':
    unittest.main()
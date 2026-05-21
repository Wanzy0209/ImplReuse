import torch
import torch.special
import unittest

class TestLogSumExpCompileRegression(unittest.TestCase):
    def test_logsumexp_dim0_large_tensor(self):
        """
        Test case based on Issue 164301 (torch.compile regression for mxfp8 quantization).
        
        The original issue reports a performance regression in torch.compile for operations
        along dimension 0 (rows) on large tensors (M=16384, K=16384). 
        torch.special.logsumexp is identified as a similar API due to shared reduction 
        patterns and codegen logic in inductor.
        
        This test verifies that torch.compile correctly handles logsumexp along dim 0
        for large tensors, ensuring the compilation path remains valid.
        """
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping GPU test")

        # Reproduce the tensor dimensions from the bug report
        M, K = 16384, 16384
        # Use float32 as standard input for logsumexp
        x = torch.randn(M, K, device='cuda', dtype=torch.float32)

        def func(inp):
            # The bug specifically highlights 'dim0' operations (row-wise)
            return torch.special.logsumexp(inp, dim=0)

        # Compile the function using torch.compile
        compiled_func = torch.compile(func)

        # Run eager execution for baseline correctness
        expected = func(x)

        # Run compiled execution
        # Warmup run to trigger compilation and potential codegen issues
        compiled_func(x)
        actual = compiled_func(x)

        # Verify correctness to ensure the codegen path is valid and not regressed
        self.assertTrue(torch.allclose(expected, actual, rtol=1e-4, atol=1e-5))

if __name__ == '__main__':
    unittest.main()
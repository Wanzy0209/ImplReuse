import torch
import torch.special
import unittest

class TestScaledModifiedBesselK0Compile(unittest.TestCase):
    def test_compile_correctness(self):
        """
        Regression test for torch.compile involving torch.special.scaled_modified_bessel_k0.
        This test mirrors the logic of Issue 164301, where torch.compile performance
        and correctness were evaluated on a specific operation using large tensors.
        """
        # The original issue was specific to CUDA (B200 GPUs)
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping test")

        # Use a large tensor size similar to the bug report (M=16384, K=16384)
        # to ensure the inductor backend is engaged and potential performance regressions
        # or codegen errors are surfaced.
        x = torch.randn(16384, 16384, device='cuda', dtype=torch.float32)

        # Define the function using the similar API
        def func(x):
            return torch.special.scaled_modified_bessel_k0(x)

        # Compile the function using torch.compile
        compiled_func = torch.compile(func)

        # Run eager execution to get baseline
        expected = func(x)

        # Run compiled execution
        # Running twice to account for potential warmup/compilation overhead in the first run
        _ = compiled_func(x)
        actual = compiled_func(x)

        # Verify correctness. While the original issue was a performance regression,
        # functional correctness is the primary requirement for a regression test here.
        self.assertTrue(torch.allclose(expected, actual, rtol=1e-4, atol=1e-4))

if __name__ == '__main__':
    unittest.main()
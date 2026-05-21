import torch
import unittest

class TestTorchAnyCompile(unittest.TestCase):
    def test_torch_any_to_cpu(self):
        """
        Test case adapted from Issue 168181 to verify torch.any correctness
        with torch.compile and .cpu() transfer.
        """
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        def f(x):
            # Use torch.any on GPU (replacing the user-defined Triton kernel)
            out = torch.any(x > 0.5)
            # Move to CPU and perform operation (checking synchronization)
            out_cpu = out.cpu() + 1
            return out_cpu

        x = torch.randn(4, 4, device="cuda")
        eager_out = f(x)
        compiled_out = torch.compile(f)(x)
        self.assertEqual(compiled_out, eager_out)

if __name__ == "__main__":
    unittest.main()
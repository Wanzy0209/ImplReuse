import torch
import unittest

# Check if torch.compile is available (introduced in PyTorch 2.0)
HAS_TORCH_COMPILE = hasattr(torch, 'compile')

@unittest.skipIf(not HAS_TORCH_COMPILE, "torch.compile requires PyTorch 2.0+")
class TestTorchBmmCompile(unittest.TestCase):
    """
    Test case for Issue 165892: torch.bmm + torch.compile with out_dtype.
    
    The bug describes an AssertionError when using torch.bmm with the out_dtype 
    argument inside a torch.compile decorated function on CUDA.
    """

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_bmm_out_dtype_compilation(self):
        # Setup inputs matching the bug report
        A = torch.rand((1, 1024, 1024), device="cuda", dtype=torch.float16)
        B = torch.rand((1, 1024, 1024), device="cuda", dtype=torch.float16)

        # Define the compiled function with the specific argument causing the issue
        @torch.compile
        def linear(weight, input):
            return torch.bmm(input, weight, out_dtype=torch.float32)

        # Execute the function. 
        # In the buggy version, this raises:
        # torch._inductor.exc.InductorError: AssertionError: out_dtype is not supported for Triton
        try:
            result = linear(A, B)
        except AssertionError as e:
            if "out_dtype is not supported for Triton" in str(e):
                self.fail("torch.bmm with out_dtype failed inside torch.compile (Bug 165892)")
            else:
                raise

        # Verify the output properties if the bug is fixed
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.shape, (1, 1024, 1024))

if __name__ == "__main__":
    unittest.main()
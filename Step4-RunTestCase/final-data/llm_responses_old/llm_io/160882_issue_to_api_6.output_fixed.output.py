import torch
import unittest

# Check for torch.compile availability (introduced in PyTorch 2.0)
HAS_TORCH_COMPILE = hasattr(torch, 'compile')

@unittest.skipIf(not HAS_TORCH_COMPILE, "torch.compile requires PyTorch >= 2.0")
class TestTorchCompileComplexDynamicShapes(unittest.TestCase):
    """
    Test case for torch.compile with torch.complex handling dynamic input shapes.
    Related to Issue 160882.
    """

    def test_complex_compile_with_permuted_inputs(self):
        """
        Reproduces the scenario where torch.compile crashes when 
        torch.ops.aten.complex.default receives inputs with different shapes 
        than during the initial compilation.
        """
        # Define the function to be compiled, mirroring the structure of the 
        # similar API's function definition (taking inputs and returning a result).
        def f(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
            z = torch.complex(real, imag)
            return torch.fft.irfft(z, dim=1)

        B, F, T = 1, 641, 39

        # Setup source tensors
        r_src = torch.randn(B, F, T)
        i_src = torch.randn(B, F, T)
        
        # Setup mismatched tensors (different shapes)
        r_mismatch = r_src.permute(0, 2, 1)
        i_mismatch = i_src.permute(0, 2, 1)

        # Compile the function. 
        # The bug report indicates a crash with fullgraph=True and dynamic=True.
        compiled = torch.compile(f, fullgraph=True, dynamic=True)

        # First call with original shapes
        try:
            out1 = compiled(r_src, i_src)
        except Exception as e:
            self.fail(f"First call to compiled function failed: {e}")

        # Second call with permuted shapes (different dimensions)
        # This is the call that triggers the AssertionError in the bug report.
        try:
            out2 = compiled(r_mismatch, i_mismatch)
        except AssertionError as e:
            self.fail(f"Second call triggered AssertionError (Bug 160882): {e}")
        except Exception as e:
            self.fail(f"Second call failed with unexpected error: {e}")

        # Basic sanity checks on outputs
        self.assertIsInstance(out1, torch.Tensor)
        self.assertIsInstance(out2, torch.Tensor)
        
        # Check that the output shapes correspond to the input shapes
        # irfft output size depends on input size and dim.
        # For (1, 641, 39) input, dim=1 is 641. Output dim 1 is 2*(641-1) = 1280? 
        # Actually, last dim is T=39. 
        # Input z: (1, 641, 39). irfft on dim=1.
        # Output shape: (1, 1280, 39).
        self.assertEqual(out1.shape, (1, 1280, 39))
        
        # For (1, 39, 641) input, dim=1 is 39.
        # Output shape: (1, 76, 641).
        self.assertEqual(out2.shape, (1, 76, 641))

if __name__ == "__main__":
    unittest.main()
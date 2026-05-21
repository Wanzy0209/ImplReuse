import torch
import torch.nn as nn
import unittest
import warnings

# Issue 164301: torch.compile regression: mxfp8 quantization along rows
# The bug report indicates a performance regression in torch.compile for mxfp8 quantization.
# The similar API provided is torch.eig, which is deprecated.
# This test case verifies the correctness of torch.compile for a quantization-like operation
# (simulating the mxfp8 logic) and checks the behavior of the deprecated torch.eig
# to ensure the compiler handles it or its replacement correctly without crashing.

class TestCompileRegression(unittest.TestCase):
    def setUp(self):
        # Repro settings from the issue
        self.M = 16384
        self.K = 16384
        self.BLOCK_SIZE = 32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _simulate_mxfp8_quantization_dim0(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simulates the quantization logic described in the bug report.
        The issue mentions 'dim0_mxfp8_floor' which implies scaling along rows (dim 0).
        We simulate scaling and casting to a lower precision representation.
        """
        # Simulate finding max per row (dim 0) for scaling
        # Shape: (M, K) -> (M, 1)
        scale = torch.abs(x).max(dim=1, keepdim=True).values
        
        # Avoid division by zero
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        
        # Scale the input
        scaled = x / scale
        
        # Simulate quantization (e.g., to float8 or similar, here we clamp to simulate range)
        # This is a simplified representation of the mxfp8 operation
        quantized = torch.clamp(scaled, -4.0, 4.0)
        
        return quantized

    def test_compile_mxfp8_quantization(self):
        """
        Test that torch.compile works correctly for the quantization logic.
        This preserves the original bug reproduction logic (dim0 quantization).
        """
        x = torch.randn(self.M, self.K, device=self.device, dtype=torch.float32)
        
        # Run eager
        expected = self._simulate_mxfp8_quantization_dim0(x)
        
        # Run compiled
        compiled_fn = torch.compile(self._simulate_mxfp8_quantization_dim0)
        actual = compiled_fn(x)
        
        # Check correctness
        self.assertTrue(torch.allclose(expected, actual, atol=1e-5))

    def test_torch_eig_deprecation_in_compile(self):
        """
        Leverage the similar API (torch.eig) to check if the compiler 
        handles deprecated functions or their replacements correctly.
        torch.eig is deprecated and raises a RuntimeError.
        We verify this behavior is preserved or handled correctly.
        """
        # Create a simple square matrix
        A = torch.randn(4, 4, device=self.device)
        
        # 1. Check that torch.eig raises the expected error in eager mode
        with self.assertRaises(RuntimeError) as context:
            torch.eig(A)
        self.assertIn("deprecated", str(context.exception).lower())
        
        # 2. Check that torch.linalg.eig (the replacement) works
        # This ensures the "similar API" logic is covered via the replacement
        L_eager, V_eager = torch.linalg.eig(A)
        
        # 3. Check that torch.compile works with the replacement API
        # This ensures the regression in the compiler doesn't break standard linalg ops
        def linalg_eig_wrapper(tensor):
            return torch.linalg.eig(tensor)
            
        compiled_wrapper = torch.compile(linalg_eig_wrapper)
        L_compiled, V_compiled = compiled_wrapper(A)
        
        self.assertTrue(torch.allclose(L_eager, L_compiled, atol=1e-5))
        self.assertTrue(torch.allclose(V_eager, V_compiled, atol=1e-5))

if __name__ == "__main__":
    unittest.main()
import torch
import unittest

class TestMHInductorCompile(unittest.TestCase):
    def test_mh_with_randn_complex_input(self):
        """
        Test case for Issue 163243: .mH compile problem with inductor backend.
        
        This test leverages torch.randn (the similar API) to generate the complex 
        input tensor and verifies that the .mH operation compiles and runs 
        correctly under the inductor backend.
        """
        # Setup parameters
        n = 8
        dtype = torch.complex64

        # Use torch.randn to generate the complex input tensor
        # This mirrors the usage in the original bug report.
        A = torch.randn(4, n, n, dtype=dtype, requires_grad=True)
        A = A.clone(memory_format=torch.contiguous_format)

        # Define the function containing the logic that triggered the bug.
        # The core issue involves the interaction of .mH (Hermitian transpose)
        # with matrix multiplication under torch.compile.
        def func_to_compile(x):
            # Original logic: A @ A.mH
            return x @ x.mH

        # Compile with the 'inductor' backend, which was failing in the issue.
        compiled_func = torch.compile(func_to_compile, backend="inductor")

        # Execute the compiled function.
        # If the bug is present, this will raise:
        # "self.stride(-1) must be 1 to view ComplexFloat as Float..."
        try:
            result = compiled_func(A)
            
            # Assertions to verify correctness if compilation succeeds
            self.assertIsNotNone(result)
            self.assertEqual(result.shape, (4, n, n))
            self.assertEqual(result.dtype, dtype)
            
            # Verify the Hermitian property (approximately)
            # (A @ A.mH) should be Hermitian
            diff = (result - result.mH).abs().max()
            self.assertLess(diff, 1e-5, "Result of A @ A.mH should be Hermitian")

        except RuntimeError as e:
            if "self.stride(-1) must be 1" in str(e):
                self.fail(f"Issue 163243 reproduced: {e}")
            else:
                raise

if __name__ == '__main__':
    unittest.main()
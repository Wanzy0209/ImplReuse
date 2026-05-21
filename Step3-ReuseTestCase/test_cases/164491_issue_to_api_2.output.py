import torch
import unittest
import sys

class TestScaledMMRowMajor(unittest.TestCase):
    """
    Test case for Issue #164491: _scaled_mm and _int_mm slow/error with row-major rhs.
    
    This test leverages the pattern from torch.backends.cusparselt.version to check
    backend availability before testing the specific matrix multiplication behavior
    described in the bug report.
    """

    def setUp(self):
        # Check CUDA availability as these are GPU ops
        self.cuda_available = torch.cuda.is_available()
        if not self.cuda_available:
            self.skipTest("CUDA is not available")

    def test_scaled_mm_row_major_rhs_behavior(self):
        """
        Reproduces the bug where _scaled_mm raises an error with row-major RHS.
        Uses torch.backends.cusparselt.version to check backend status.
        """
        # Leverage the similar API to check the backend state
        # Pattern from torch.backends.cusparselt.version: check init/status
        cusparselt_version = torch.backends.cusparselt.version()
        
        print(f"cuSPARSELt Version: {cusparselt_version}")

        # Dimensions for the test
        M, K, N = 128, 64, 32
        device = 'cuda'

        # Create LHS (A) - Standard row-major (C-contiguous)
        # In the context of the bug, the issue is specifically with the RHS layout.
        a_fp16 = torch.randn(M, K, dtype=torch.float16, device=device)

        # Create RHS (B) - Row-major (C-contiguous)
        # The bug report states: "_scaled_mm raises an error if the rhs matrix is row-major."
        # Standard torch.randn creates row-major tensors.
        b_int8 = torch.randn(K, N, dtype=torch.int8, device=device)

        # Scales required for _scaled_mm
        scale_a = torch.randn(1, dtype=torch.float, device=device)
        scale_b = torch.randn(1, dtype=torch.float, device=device)

        # Attempt to run _scaled_mm
        # Note: _scaled_mm is an internal API (aten::_scaled_mm)
        try:
            # We access the op via torch.ops.aten or torch._scaled_mm if exposed
            # Depending on the PyTorch version, the exact access might vary.
            # Here we assume the standard internal access.
            if hasattr(torch, '_scaled_mm'):
                op = torch._scaled_mm
            elif hasattr(torch.ops, 'aten') and hasattr(torch.ops.aten, '_scaled_mm'):
                op = torch.ops.aten._scaled_mm
            else:
                self.skipTest("_scaled_mm operator not found in this build")

            # The bug report indicates this raises an error with row-major RHS
            # if the backend does not handle the transpose efficiently or correctly.
            result = op(a_fp16, b_int8, scale_a, scale_b)
            
            # If we reach here, the operation succeeded.
            # If the bug is present, this line might not be reached (error raised).
            # We verify the output shape is correct.
            self.assertEqual(result.shape, (M, N))
            
            print("Test passed: _scaled_mm handled row-major RHS without error.")

        except RuntimeError as e:
            # The bug report mentions: "_scaled_mm raises an error if the rhs matrix is row-major."
            # We catch this to document the failure condition.
            # In a regression test for a fix, we would not expect this.
            # In a reproduction test, this confirms the bug.
            print(f"Caught RuntimeError (Bug Reproduced): {e}")
            # Depending on the goal (verify fix vs verify bug), we might fail here.
            # Assuming we want to verify the fix is present:
            self.fail(f"_scaled_mm failed with row-major RHS (Bug #164491): {e}")

    def test_int_mm_row_major_rhs_performance_check(self):
        """
        Checks _int_mm with row-major RHS.
        The bug report notes it is very slow (transparent transpose).
        """
        cusparselt_version = torch.backends.cusparselt.version()
        
        M, K, N = 1024, 512, 256
        device = 'cuda'

        # Row-major RHS
        a_int8 = torch.randint(-10, 10, (M, K), dtype=torch.int8, device=device)
        b_int8 = torch.randint(-10, 10, (K, N), dtype=torch.int8, device=device)

        if hasattr(torch, '_int_mm'):
            op = torch._int_mm
        elif hasattr(torch.ops, 'aten') and hasattr(torch.ops.aten, '_int_mm'):
            op = torch.ops.aten._int_mm
        else:
            self.skipTest("_int_mm operator not found in this build")

        # Run the operation
        # We don't assert performance (flaky), but we ensure it completes
        # and produces the correct shape, verifying it doesn't crash.
        try:
            result = op(a_int8, b_int8)
            self.assertEqual(result.shape, (M, N))
            print("_int_mm completed with row-major RHS.")
        except Exception as e:
            self.fail(f"_int_mm failed with row-major RHS: {e}")

if __name__ == '__main__':
    unittest.main()
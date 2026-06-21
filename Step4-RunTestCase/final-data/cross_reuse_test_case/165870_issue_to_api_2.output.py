import torch
import unittest

class TestLUFactorSingularMatrix(unittest.TestCase):
    """
    Test case for torch.linalg.lu_factor on singular matrices using the MPS backend.
    
    This test reflects the structural pattern of the similar API 
    (tf.compat.v1.local_variables_initializer) by checking the execution context 
    (device availability) before proceeding with the operation.
    """

    def test_lu_factor_mps_singular_matrix_error(self):
        """
        Verifies that torch.linalg.lu_factor raises a RuntimeError on a singular matrix
        when using the MPS device, matching the CPU behavior.
        """
        # Check if MPS is available, similar to checking context.executing_eagerly()
        # in the similar API to determine execution path.
        if not torch.backends.mps.is_available():
            self.skipTest("MPS backend is not available. Skipping MPS-specific test.")

        # Create a singular matrix (determinant is 0)
        # [[1, 2], [2, 4]] -> Row 2 is 2 * Row 1
        t = torch.tensor([[1.0, 2.0], [2.0, 4.0]], device="mps")

        # Expect a RuntimeError similar to the CPU behavior described in the bug report.
        # The error message indicates a zero pivot was encountered.
        with self.assertRaises(RuntimeError) as context:
            torch.linalg.lu_factor(t)

        self.assertTrue("is zero" in str(context.exception) or "singular" in str(context.exception).lower(),
                        "Expected error message indicating a zero pivot or singular matrix.")

if __name__ == "__main__":
    unittest.main()
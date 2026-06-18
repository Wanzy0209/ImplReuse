import torch
import unittest

class TestLUFactorSingularMatrix(unittest.TestCase):
    """
    Test case to verify that torch.linalg.lu_factor raises a RuntimeError
    for singular matrices on both CPU and MPS devices.
    
    This addresses the issue where MPS backend failed to raise an error
    unlike the CPU backend.
    """

    def test_cpu_singular_matrix_raises(self):
        """Test that lu_factor raises RuntimeError on CPU for singular matrix."""
        t = torch.tensor([[1.0, 2.0], [2.0, 4.0]])
        
        with self.assertRaises(RuntimeError) as context:
            torch.linalg.lu_factor(t)
        
        self.assertIn("U[2,2] is zero", str(context.exception))

    @unittest.skipIf(not torch.backends.mps.is_available(), "MPS not available")
    def test_mps_singular_matrix_raises(self):
        """Test that lu_factor raises RuntimeError on MPS for singular matrix."""
        t = torch.tensor([[1.0, 2.0], [2.0, 4.0]], device="mps")
        
        # This assertion ensures the MPS behavior matches CPU behavior
        with self.assertRaises(RuntimeError) as context:
            torch.linalg.lu_factor(t)
            
        self.assertIn("U[2,2] is zero", str(context.exception))

if __name__ == "__main__":
    unittest.main()
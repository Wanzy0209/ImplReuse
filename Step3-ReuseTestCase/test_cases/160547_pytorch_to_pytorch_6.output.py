import torch
import unittest

class TestPcaLowrankNamedTuple(unittest.TestCase):
    def test_pca_lowrank_namedtuple_output(self):
        """
        Test that torch.pca_lowrank returns a NamedTuple and that
        the fields are accessible, similar to the NamedTuple context
        in the original bug report.
        """
        # Create a random matrix
        A = torch.randn(5, 3)
        
        # Call pca_lowrank
        # The function returns a namedtuple (U, S, V)
        result = torch.pca_lowrank(A)
        
        # Verify the result has the expected fields of a namedtuple
        self.assertTrue(hasattr(result, 'U'))
        self.assertTrue(hasattr(result, 'S'))
        self.assertTrue(hasattr(result, 'V'))
        
        # Verify unpacking works
        U, S, V = result
        
        # Check shapes
        self.assertEqual(U.shape, (5, 3))
        self.assertEqual(S.shape, (3,))
        self.assertEqual(V.shape, (3, 3))
        
        # Verify values are reasonable (PCA property)
        # A approx U @ diag(S) @ V.t()
        reconstructed = U @ torch.diag(S) @ V.t()
        self.assertTrue(torch.allclose(A, reconstructed, atol=1e-2))

if __name__ == '__main__':
    unittest.main()
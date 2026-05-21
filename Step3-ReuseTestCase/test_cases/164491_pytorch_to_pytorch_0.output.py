import torch
import unittest

class TestMatmulRowMajorRHS(unittest.TestCase):
    """
    Test case for torch.matmul handling row-major right-hand side (RHS) matrices.
    
    Related to Issue #164491:
    The internal APIs _scaled_mm and _int_mm were reported to raise errors or 
    perform poorly when the RHS matrix is row-major. This test verifies that 
    the standard torch.matmul API handles this scenario correctly and robustly.
    """

    def test_matmul_row_major_rhs(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Define dimensions
        M, K, N = 128, 64, 32
        
        # Create LHS (M, K)
        lhs = torch.randn(M, K, device=device)
        
        # Create RHS (K, N) in Row-Major layout.
        # In PyTorch, a default tensor of shape (K, N) is stored row-major (C-contiguous).
        rhs_row_major = torch.randn(K, N, device=device)
        
        # Verify the layout is indeed row-major (contiguous)
        self.assertTrue(rhs_row_major.is_contiguous())

        # Perform matrix multiplication.
        # Unlike _scaled_mm (which raised an error in the bug report), 
        # torch.matmul should handle this seamlessly.
        result = torch.matmul(lhs, rhs_row_major)
        
        # Verify output shape
        self.assertEqual(result.shape, (M, N))
        
        # Verify correctness by comparing with torch.mm
        expected = torch.mm(lhs, rhs_row_major)
        self.assertTrue(torch.allclose(result, expected))

    def test_matmul_column_major_rhs(self):
        """
        Test torch.matmul with a column-major RHS for completeness.
        In PyTorch, a transposed contiguous tensor becomes column-major (non-contiguous).
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        M, K, N = 128, 64, 32
        
        lhs = torch.randn(M, K, device=device)
        
        # Create a tensor that is column-major (non-contiguous)
        # by transposing a row-major tensor.
        temp = torch.randn(N, K, device=device)
        rhs_col_major = temp.t() # Shape (K, N), strides (1, N)
        
        self.assertFalse(rhs_col_major.is_contiguous())
        
        result = torch.matmul(lhs, rhs_col_major)
        self.assertEqual(result.shape, (M, N))
        
        # Verify correctness
        expected = torch.mm(lhs, rhs_col_major)
        self.assertTrue(torch.allclose(result, expected))

if __name__ == "__main__":
    unittest.main()
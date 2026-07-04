import torch
import torch.nn as nn

class FastLearnedCellCholesky(nn.Module):
    """
    Adapted from FastLearnedCellX3 to test torch.cholesky_solve.
    Replaces the addressing/indexing logic with a linear algebra solve.
    """
    def __init__(self, N):
        super().__init__()
        # Initialize a random upper triangular matrix U
        # A = U^T U will be positive definite
        self.U = nn.Parameter(torch.triu(torch.randn(N, N)))
        # Ensure diagonal dominance for stability
        with torch.no_grad():
            self.U.data += torch.eye(N)

    def forward(self, input_tensor):
        # input_tensor acts as 'b' in Ax = b
        # self.U acts as the Cholesky factor 'u'
        # Solves Ax = b where A = U^T U
        
        # torch.cholesky_solve expects the input 'b' to be of shape (*, N, K) or (*, N)
        # where the last dimension N matches the dimension of the square matrix U (N, N).
        # If input_tensor is (batch_size, N), we treat the batch_size as K (number of right-hand sides).
        # Therefore, we transpose input_tensor to (N, batch_size) for the solve operation.
        # The result will be (N, batch_size), so we transpose it back to (batch_size, N).
        return torch.cholesky_solve(input_tensor.T, self.U, upper=True).T

def test_cholesky_solve():
    N = 4
    batch_size = 3
    
    # Instantiate the module
    model = FastLearnedCellCholesky(N)
    
    # Create input tensor B (batch_size, N)
    # Note: cholesky_solve expects input to be at least 2D
    B = torch.randn(batch_size, N)
    
    # Run the forward pass
    result = model(B)
    
    # Verify the result
    # Reconstruct A = U^T U
    U = model.U
    A = U.T @ U
    
    # Compute expected using torch.linalg.solve
    # Note: torch.linalg.solve expects A to be (..., N, N) and B to be (..., N)
    # Our B is (batch, N), so we transpose for solve and transpose back
    expected = torch.linalg.solve(A, B.T).T
    
    # Assert correctness
    assert torch.allclose(result, expected, atol=1e-5), "Cholesky solve result mismatch"
    print("Test passed successfully.")

if __name__ == "__main__":
    test_cholesky_solve()
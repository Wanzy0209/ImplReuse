import torch

# Ensure reproducibility
torch.manual_seed(0)

# Create a random matrix
A = torch.randn(10, 5)
# Fix: Set q to the minimum dimension to ensure full rank reconstruction.
# With q=3, the reconstruction is a low-rank approximation and will not match the original matrix exactly.
q = min(A.shape)

# Call the similar API: torch.pca_lowrank
U, S, V = torch.pca_lowrank(A, q=q)

# Verify correctness by reconstructing the centered matrix
# According to the docs: A_centered approx U @ diag(S) @ V.T
A_centered = A - A.mean(dim=0)
A_reconstructed = U @ torch.diag(S) @ V.T

# Assert that the reconstruction is close to the original centered matrix
torch.testing.assert_close(A_centered, A_reconstructed, rtol=1e-3, atol=1e-3)
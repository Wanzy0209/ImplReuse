import torch
import torch.nn as nn

def test_lobpcg_compile_inductor():
    """
    Test case to verify torch.lobpcg behavior when compiled 
    with the inductor backend, based on the regression reported 
    for torch.compile (Issue ID: 160084).
    """
    # Check for CUDA availability as the bug is specific to CUDA streams
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test.")
        return

    class LobpcgModel(nn.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, A):
            # Call the similar API: torch.lobpcg
            # We request k=2 eigenvalues for a 10x10 matrix
            eigenvalues, eigenvectors = torch.lobpcg(A, k=2)
            return eigenvalues

    model = LobpcgModel()
    torch_device = "cuda"
    
    # Apply torch.compile with backend="inductor" as in the original bug report
    model.forward = torch.compile(model.forward, backend="inductor")

    # Create a symmetric positive definite matrix on CUDA
    # lobpcg requires A to be symmetric positive definite
    n = 10
    X = torch.randn(n, n, device=torch_device)
    # A = X @ X.T + I ensures symmetry and positive definiteness
    A = X @ X.T + torch.eye(n, device=torch_device)

    try:
        # Run the compiled model
        result = model(A)
        
        # Basic assertion to verify execution
        assert result is not None
        assert result.shape[0] == 2
        print("Test passed: torch.lobpcg compiled with inductor backend executed successfully.")
        
    except RuntimeError as e:
        # Check for the specific internal assertion failure mentioned in the bug report
        if "opt_ready_stream && opt_parent_stream" in str(e):
            print(f"Regression detected: {e}")
        else:
            # Re-raise if it's a different error
            raise

if __name__ == "__main__":
    test_lobpcg_compile_inductor()
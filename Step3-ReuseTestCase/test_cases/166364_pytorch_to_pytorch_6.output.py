import torch
import torch.nn as nn

def test_lobpcg_with_learnable_scalar(use_compile=False):
    """
    Test torch.lobpcg with a learnable scalar parameter.
    This adapts the issue where flex_attention failed with learnable scalars
    inside the score_mod function. Here, we test if lobpcg handles inputs
    derived from learnable scalars correctly, especially with torch.compile.
    """
    # Create a learnable scalar parameter
    temp = nn.Parameter(torch.tensor(1.0))

    # Create a symmetric positive definite matrix dependent on the scalar
    # This simulates the usage of a learnable scalar in the computation graph
    base_matrix = torch.randn(5, 5)
    A = temp * (base_matrix @ base_matrix.T)

    # Initial guess for eigenvectors
    X = torch.randn(5, 2)

    def run_lobpcg(A_in, X_in):
        # lobpcg returns eigenvalues and eigenvectors
        eigenvalues, _ = torch.lobpcg(A_in, k=2, X=X_in)
        return eigenvalues.sum()

    if use_compile:
        run_lobpcg = torch.compile(run_lobpcg)

    # Forward pass
    try:
        loss = run_lobpcg(A, X)
        print(f"Forward pass (compile={use_compile}) successful. Loss: {loss.item()}")
    except Exception as e:
        print(f"Forward pass (compile={use_compile}) failed: {e}")
        raise

    # Backward pass
    try:
        loss.backward()
        print(f"Backward pass (compile={use_compile}) successful. Grad: {temp.grad}")
    except Exception as e:
        print(f"Backward pass (compile={use_compile}) failed: {e}")
        raise

    # Assertions to verify correctness
    assert temp.grad is not None, "Gradient for learnable scalar is None"
    assert not torch.isnan(temp.grad), "Gradient is NaN"
    assert not torch.isinf(temp.grad), "Gradient is Inf"

if __name__ == "__main__":
    print("Testing torch.lobpcg without compile...")
    test_lobpcg_with_learnable_scalar(use_compile=False)
    
    print("\nTesting torch.lobpcg with compile...")
    test_lobpcg_with_learnable_scalar(use_compile=True)
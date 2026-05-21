import torch
import torch.lobpcg
import time

def test_lobpcg_compile_regression():
    """
    Test case for torch.lobpcg under torch.compile.
    Adapted from the context of a torch.compile performance regression 
    observed in mxfp8 quantization (Issue ID: 164301).
    
    This test verifies that torch.lobpcg compiles and runs correctly 
    on large matrices, similar to the scale used in the original bug report.
    """
    # Dimensions similar to the original bug report (M=16384, K=16384)
    N = 16384
    k = 10  # Number of eigenvalues to compute
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Matrix Size (N): {N}")
    print(f"Eigenpairs (k): {k}")

    # Generate a symmetric positive definite matrix A
    # A = X @ X.T + I ensures positive definiteness
    torch.manual_seed(42)
    X = torch.randn(N, N, device=device, dtype=torch.float32)
    A = X @ X.T + torch.eye(N, device=device)

    # Initial guess for eigenvectors
    Y = torch.randn(N, k, device=device, dtype=torch.float32)

    # Define the operation to be compiled
    def run_lobpcg(A, Y):
        # lobpcg returns (eigenvalues, eigenvectors)
        return torch.lobpcg(A, k=k, X=Y)

    # Compile the function
    print("Compiling torch.lobpcg...")
    compiled_run_lobpcg = torch.compile(run_lobpcg)

    # Warmup run (triggers compilation)
    try:
        eigenvalues, eigenvectors = compiled_run_lobpcg(A, Y)
    except Exception as e:
        print(f"Compilation or execution failed: {e}")
        raise

    # Timed run to observe performance (mimicking the original benchmark)
    start = time.time()
    eigenvalues, eigenvectors = compiled_run_lobpcg(A, Y)
    end = time.time()

    print(f"Execution time: {end - start:.4f}s")

    # Verify results
    assert eigenvalues is not None, "Eigenvalues should not be None"
    assert eigenvectors is not None, "Eigenvectors should not be None"
    assert eigenvalues.shape[0] == k, f"Expected {k} eigenvalues, got {eigenvalues.shape[0]}"
    assert eigenvectors.shape == (N, k), f"Expected eigenvectors shape ({N}, {k}), got {eigenvectors.shape}"
    
    # Check that eigenvalues are real (since A is symmetric positive definite)
    assert torch.all(torch.isreal(eigenvalues)), "Eigenvalues should be real for SPD matrix"
    
    # Check for numerical stability
    assert torch.all(torch.isfinite(eigenvalues)), "Eigenvalues contain NaN or Inf"
    assert torch.all(torch.isfinite(eigenvectors)), "Eigenvectors contain NaN or Inf"

    print("Test passed successfully.")

if __name__ == "__main__":
    test_lobpcg_compile_regression()
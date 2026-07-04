import os
import torch

# Ensure logs are enabled if needed, mimicking the original setup
# os.environ["TORCH_LOGS"] = "output_code"

device = "cuda" if torch.cuda.is_available() else "cpu"

def test_lobpcg_compilation():
    # Check for torch.compile availability (requires PyTorch 2.0+)
    if not hasattr(torch, 'compile'):
        print("torch.compile is not available (requires PyTorch >= 2.0). Skipping test.")
        return

    # Setup a simple eigenvalue problem
    # A must be symmetric positive definite
    n = 20
    k = 5
    A = torch.randn(n, n, device=device)
    A = A @ A.T + torch.eye(n, device=device) * 0.1  # Make it SPD
    X = torch.randn(n, k, device=device)

    # 1. Run in eager mode (reference)
    print("Running eager mode...")
    w_eager, v_eager = torch.lobpcg(A, k=k, X=X)

    # 2. Run in compiled mode
    # We wrap torch.lobpcg in torch.compile to check for the miscompilation issue
    # similar to how slide_to_the_left2 was wrapped in the original bug report.
    print("Running compiled mode...")
    compiled_lobpcg = torch.compile(torch.lobpcg)

    # The original bug was a race condition that required a loop to catch.
    # We will run it multiple times to ensure stability.
    for attempt in range(1, 10):
        w_compiled, v_compiled = compiled_lobpcg(A, k=k, X=X)

        # Check if results match
        # Note: Eigenvectors can differ by sign, so we check absolute values or use a robust comparison
        if not torch.allclose(w_eager, w_compiled, rtol=1e-4, atol=1e-4):
            print(f"Mismatch found on attempt {attempt}")
            print("Eager eigenvalues:", w_eager)
            print("Compiled eigenvalues:", w_compiled)
            raise AssertionError("Eigenvalues mismatch between eager and compiled modes")

        # Check eigenvectors (A @ v = w @ v)
        # Residual check is often better for eigenvectors
        res_eager = torch.norm(A @ v_eager - v_eager @ torch.diag(w_eager))
        res_compiled = torch.norm(A @ v_compiled - v_compiled @ torch.diag(w_compiled))

        if not torch.allclose(res_eager, res_compiled, rtol=1e-3, atol=1e-3):
             print(f"Residual mismatch found on attempt {attempt}")
             raise AssertionError("Eigenvectors residual mismatch")

    print("Test passed: torch.lobpcg behaves consistently under torch.compile")

if __name__ == "__main__":
    test_lobpcg_compilation()
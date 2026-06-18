import torch

def example_function_lobpcg():
    # Adapted function using torch.lobpcg instead of custom Cholesky logic
    def compute_eigen(matrix):
        # torch.lobpcg is a linear algebra function that might involve 
        # similar internal operations (like Cholesky) as the original bug.
        # We test if it triggers the same contiguity issue when compiled on MPS.
        eigenvalues, eigenvectors = torch.lobpcg(matrix, k=1)
        return eigenvalues

    return compute_eigen

if __name__ == "__main__":
    # Check for MPS availability
    if not torch.backends.mps.is_available():
        print("MPS device is not available. Skipping test.")
    else:
        device = torch.device("mps")
        dtype = torch.float32

        # Create a symmetric positive definite matrix (required for lobpcg)
        # Using a 3x3 matrix similar to the dimensions in the original bug report
        diag_values = torch.tensor((20., 0.5, 5.), device=device, dtype=dtype)
        matrix = torch.diag(diag_values ** 2)
        # Ensure it is strictly positive definite for lobpcg
        matrix = matrix + torch.eye(3, device=device, dtype=dtype)

        # Compile the function using torch.compile
        compiled_function = torch.compile(example_function_lobpcg())

        try:
            # Run the compiled function
            res = compiled_function(matrix)
            print("Test passed. Result:", res)
            
            # Basic assertion to ensure output is valid
            assert res.shape == (1,), f"Expected shape (1,), got {res.shape}"
            assert torch.isfinite(res).all(), "Result contains NaN or Inf"
            
        except RuntimeError as e:
            if "is_contiguous()" in str(e):
                print(f"Bug reproduced: {e}")
            else:
                raise
import torch
import torch._dynamo

# Test case for torch.lobpcg within a torch.compile(fullgraph=True) context
# involving torch.compiler.disable to verify interaction behavior.

def test_lobpcg_with_compile_fullgraph_and_disable():
    """
    Verifies that torch.lobpcg can be used inside a function compiled with
    fullgraph=True, even when parts of the computation are explicitly
    disabled using torch.compiler.disable.
    """
    
    # Define a helper function that we intend to disable
    @torch.compiler.disable
    def helper_fn(x):
        return x + 1

    # Define the main function to be compiled
    def fn(A):
        # Perform a disabled operation
        A_shifted = helper_fn(A)
        
        # Call torch.lobpcg (the similar API)
        # We use a simple setup for lobpcg: finding eigenvalues of A_shifted
        # A must be symmetric positive definite for lobpcg
        eigenvalues, _ = torch.lobpcg(A_shifted, k=1)
        
        return eigenvalues

    # Create a symmetric positive definite matrix
    A = torch.randn(5, 5)
    A = A @ A.T + 1e-3 * torch.eye(5)

    # Compile with fullgraph=True
    # The bug report indicates that this combination (fullgraph=True + disable)
    # might raise torch._dynamo.exc.Unsupported.
    # This test checks if torch.lobpcg behaves correctly in this scenario.
    try:
        compiled_fn = torch.compile(fn, fullgraph=True)
        result = compiled_fn(A)
        
        # Basic assertion to ensure execution
        assert result.shape == (1,), f"Expected shape (1,), got {result.shape}"
        print("Test passed: torch.lobpcg works with fullgraph=True and torch.compiler.disable")
        
    except torch._dynamo.exc.Unsupported as e:
        # If this specific error occurs, it mirrors the original bug report's issue
        # but in the context of the similar API (lobpcg) being the workload.
        print(f"Test failed with Unsupported error: {e}")
        raise

if __name__ == "__main__":
    test_lobpcg_with_compile_fullgraph_and_disable()
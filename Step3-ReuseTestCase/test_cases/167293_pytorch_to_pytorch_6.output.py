import torch
import torch._dynamo
from torch.export import export, Dim

def test_pca_lowrank_dynamic_export():
    """
    Test that torch.pca_lowrank can be exported with dynamic shapes
    without triggering torch._dynamo.exc.UserError: Constraints violated.
    
    This test is derived from Issue 167293, where torch.export.export failed
    with constraint violations on dynamic sequences. We adapt the scenario
    to verify the similar API torch.pca_lowrank.
    """
    
    # Define a function using torch.pca_lowrank
    def pca_model(A):
        # Perform PCA. q is the number of principal components.
        # We return the projection U.
        U, S, V = torch.pca_lowrank(A, q=2, center=True)
        return U

    # Define dynamic dimensions for the input tensor A
    # The bug report involved a 'seq' dimension constraint violation.
    # We define a dynamic dimension 'batch' to simulate dynamic input sizes.
    batch_dim = Dim("batch", min=1, max=100)
    feat_dim = Dim("feat", min=1, max=100)

    # Create an example input (e.g., 10 samples, 5 features)
    example_input = torch.randn(10, 5)

    try:
        # Attempt to export the model with dynamic shapes
        # This mimics the export process that failed in the original bug report
        exported_model = export(
            pca_model,
            (example_input,),
            dynamic_shapes=({"A": {0: batch_dim, 1: feat_dim}},)
        )

        # Verify the export by running with a different input size
        # Original was 10, we try 20 (within the specified constraint max=100)
        new_input = torch.randn(20, 5)
        result = exported_model(new_input)

        # Assertions to verify correctness
        # U should have shape (m, k) where m is batch size and k is q (2)
        assert result.shape == (20, 2), f"Expected shape (20, 2), got {result.shape}"
        
        print("Test Passed: torch.pca_lowrank exported and executed successfully with dynamic shapes.")

    except torch._dynamo.exc.UserError as e:
        # Catch the specific error mentioned in the bug report
        print(f"Test Failed with UserError: {e}")
        raise
    except Exception as e:
        print(f"Test Failed with unexpected error: {e}")
        raise

if __name__ == "__main__":
    test_pca_lowrank_dynamic_export()
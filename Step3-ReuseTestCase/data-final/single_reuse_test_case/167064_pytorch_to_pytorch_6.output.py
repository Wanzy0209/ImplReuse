import torch
import torch.distributions

# Test case adapted from Issue 167064
# Original issue: torch.compile (via unused code) calls
# torch.distributions.Distribution.set_default_validate_args(False)
# This test verifies if the similar API torch.lobpcg exhibits the same side effect.

def test_lobpcg_distribution_side_effect():
    # Setup: Create a symmetric positive definite matrix for lobpcg
    torch.manual_seed(42)
    A = torch.randn(10, 10)
    A = A @ A.T + 5 * torch.eye(10)  # Ensure positive definiteness
    k = 3

    # Pre-condition: Set distribution validation to True
    torch.distributions.Distribution.set_default_validate_args(True)

    # Action: Call the similar API (torch.lobpcg)
    # lobpcg returns (eigenvalues, eigenvectors)
    try:
        eigenvalues, _ = torch.lobpcg(A, k=k)
    except RuntimeError:
        # lobpcg might fail to converge on random matrices, 
        # but we are interested in the side effect during execution.
        pass

    # Verification: Check if the global state was changed to False
    # The bug report indicates torch.compile sets this to False.
    # We check the internal attribute _validate_args to ensure it remains True.
    assert torch.distributions.Distribution._validate_args == True, \
        "torch.lobpcg incorrectly modified global distribution validation state to False"

if __name__ == "__main__":
    test_lobpcg_distribution_side_effect()
    print("Test passed: torch.lobpcg did not affect distribution validation args.")
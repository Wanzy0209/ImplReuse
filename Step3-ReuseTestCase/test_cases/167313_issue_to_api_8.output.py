import torch
import torch.nn.functional as F

def test_addmm_alpha_beta_compilation():
    """
    Test case to verify that torch.compile preserves alpha and beta parameters
    in torch.addmm operations, especially when followed by pointwise operations.
    
    This test is adapted from the bug report where the Inductor backend was 
    replacing addmm with add(mm), effectively ignoring alpha and beta.
    """
    # Check for CUDA availability as the original bug was reported on CUDA
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test.")
        return

    device = "cuda"
    
    # Setup inputs
    x = torch.rand(2, device=device)
    a = torch.rand(2, 3, device=device)
    b = torch.rand(3, 2, device=device)

    # Define the function with non-default alpha and beta
    # The bug occurs specifically when addmm is followed by a pointwise op (like relu)
    def func(x, a, b):
        return F.relu(torch.addmm(x, a, b, alpha=0.5, beta=0.5))

    # Compile the function
    compiled_func = torch.compile(func)

    # Execute both eager and compiled versions
    expected = func(x, a, b)
    actual = compiled_func(x, a, b)

    # Verify the results match
    # The bug causes alpha/beta to be treated as 1.0, resulting in values ~2x larger
    # than expected (since 0.5 vs 1.0).
    assert torch.allclose(expected, actual, atol=1e-5, rtol=1e-5), (
        f"Mismatch found.\nExpected (eager):\n{expected}\n"
        f"Actual (compiled):\n{actual}\n"
        "This indicates that alpha/beta parameters were ignored during compilation."
    )
    
    print("Test passed: alpha and beta parameters were correctly preserved.")

if __name__ == "__main__":
    test_addmm_alpha_beta_compilation()
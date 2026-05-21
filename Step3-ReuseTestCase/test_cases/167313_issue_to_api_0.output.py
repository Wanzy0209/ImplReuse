import torch
import torch.nn.functional as F

def test_addmm_alpha_beta_with_compile():
    """
    Test case to verify that torch.compile correctly handles alpha and beta 
    parameters in torch.addmm when followed by a pointwise operation.
    
    This test preserves the logic of the original bug report (Issue 167313) 
    while incorporating input validation patterns similar to the 
    tf.keras.metrics.sparse_categorical_accuracy API (checking tensor ranks).
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create inputs
    x = torch.rand(2, device=device)
    a = torch.rand(2, 3, device=device)
    b = torch.rand(3, 2, device=device)

    # Leverage pattern from similar API: Check ranks/shapes before operation
    # Similar to how sparse_categorical_accuracy checks y_pred_rank and y_true_rank
    if x.dim() not in [1, 2]:
        raise ValueError(f"Input x must be 1D or 2D, got {x.dim()}D")
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError("Matrices a and b must be 2D")
    if a.size(1) != b.size(0):
        raise ValueError(f"Matrix multiplication dimension mismatch: a size {a.shape}, b size {b.shape}")

    # Define scaling factors
    alpha = 0.5
    beta = 0.5

    # Original bug reproduction logic: addmm followed by relu
    # The bug occurred because the compiler replaced addmm with add(mm) 
    # and ignored alpha/beta, effectively treating them as 1.0.
    f = lambda x, a, b: F.relu(torch.addmm(x, a, b, alpha=alpha, beta=beta))

    # Compile the function
    fc = torch.compile(f)

    # Execute eager and compiled versions
    res_eager = f(x, a, b)
    res_compiled = fc(x, a, b)

    # Assert that the results are close
    # If the bug exists, res_compiled will be significantly different 
    # (specifically, roughly double if alpha=beta=0.5 vs default 1.0)
    assert torch.allclose(res_eager, res_compiled, atol=1e-4), (
        f"Mismatch detected between eager and compiled results.\n"
        f"Eager result:\n{res_eager}\n"
        f"Compiled result:\n{res_compiled}\n"
        f"This indicates that alpha/beta parameters might be ignored during compilation."
    )

if __name__ == "__main__":
    test_addmm_alpha_beta_with_compile()
    print("Test passed successfully.")
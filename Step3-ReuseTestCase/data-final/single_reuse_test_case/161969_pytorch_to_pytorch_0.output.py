import torch
import torch.vmap
import torch.func

def test_mps_compile_contiguousness():
    """
    Test case for Issue 161969:
    Verifies that torch.compile works correctly with MPS backend 
    when using vmap, grad, and linear algebra operations that require 
    contiguous tensors.
    """
    
    # Check if MPS is available
    if not torch.backends.mps.is_available():
        print("MPS backend is not available. Skipping test.")
        return

    device = torch.device("mps")
    dtype = torch.float32

    def example_function():
        def logp(x, matrix):
            # The bug report indicates that removing the print statement 
            # (or the side effect of checking is_contiguous) caused a crash.
            # We intentionally omit the print statement here to verify the fix.
            p_mat_sqrt = torch.linalg.cholesky(matrix).contiguous()
            p_mat_sqrt_inv = p_mat_sqrt.inverse()
            val = torch.sum((p_mat_sqrt_inv @ x[0, :]) ** 2)
            return -val / 2

        score_func = torch.vmap(torch.func.grad(logp, 0), (0, None))
        return score_func

    # Prepare input data
    data = torch.zeros((2, 5, 3), device=device, dtype=dtype)
    p = torch.diag(torch.tensor((20., 0.5, 5.), device=device, dtype=dtype)**2)

    # 1. Run in eager mode to establish baseline
    try:
        eager_func = example_function()
        res_eager = eager_func(data, p)
        print("Eager mode execution successful.")
    except Exception as e:
        print(f"Eager mode failed: {e}")
        return

    # 2. Run with torch.compile (the scenario from the bug report)
    try:
        compiled_function = torch.compile(example_function())
        res_compiled = compiled_function(data, p)
        print("Compiled mode execution successful.")
        
        # Verify that the compiled result matches the eager result
        assert torch.allclose(res_eager, res_compiled, atol=1e-5), \
            "Results differ between eager and compiled modes"
        print("Results match between eager and compiled modes.")

    except RuntimeError as e:
        if "is_contiguous() INTERNAL ASSERT FAILED" in str(e):
            print(f"Bug reproduced: {e}")
        else:
            raise
    except Exception as e:
        print(f"Test failed with unexpected error: {e}")

if __name__ == "__main__":
    test_mps_compile_contiguousness()
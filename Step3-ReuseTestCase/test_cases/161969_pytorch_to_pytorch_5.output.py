import torch

def test_torch_any_mps_compile():
    """
    Test case for torch.any on MPS device within a compiled function.
    Adapted from the bug report context to verify torch.any behavior.
    """
    def example_function():
        def logp(x, matrix):
            # Adaptation: Use torch.any to check a condition on the matrix.
            # This replaces the print statement from the original bug report
            # with a functional API call (torch.any) to verify its behavior
            # in the MPS compilation context.
            any_check = torch.any(matrix > 0)

            p_mat_sqrt = torch.linalg.cholesky(matrix).contiguous()
            p_mat_sqrt_inv = p_mat_sqrt.inverse()
            val = torch.sum((p_mat_sqrt_inv @ x[0, :]) ** 2)
            
            # Include the result of torch.any in the return value to ensure
            # it is part of the computation graph.
            return -val/2 + float(any_check)

        score_func = torch.vmap(torch.func.grad(logp, 0), (0, None))
        return score_func

    if __name__ == "__main__":
        # Check if MPS is available
        if not torch.backends.mps.is_available():
            print("MPS device not available. Skipping test.")
            return

        device = torch.device("mps")
        dtype = torch.float32
        data = torch.zeros((2, 5, 3), device=device, dtype=dtype)
        
        # Compile the function
        compiled_function = torch.compile(example_function())

        p = torch.diag(torch.tensor((20., 0.5, 5,), device=device, dtype=dtype)**2)
        
        try:
            res = compiled_function(data, p)
            print(f"Test passed. Result shape: {res.shape}")
        except RuntimeError as e:
            print(f"Test failed with RuntimeError: {e}")

if __name__ == "__main__":
    test_torch_any_mps_compile()
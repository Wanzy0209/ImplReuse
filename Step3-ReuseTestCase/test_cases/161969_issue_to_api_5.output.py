import torch

def test_mps_compile_contiguity_with_moving_avg():
    """
    Test case for Issue 161969 adapted with logic from tf.keras.backend.moving_average_update.
    
    The original bug involves a RuntimeError regarding contiguity when using torch.compile
    on MPS with linear algebra operations (cholesky, inverse, matmul) inside a vmap/grad context.
    
    This test integrates the 'moving_average_update' logic (x = x * momentum + value * (1 - momentum))
    into the computation graph to verify if the compilation issue persists or is resolved.
    """
    
    # Skip if MPS is not available
    if not torch.backends.mps.is_available():
        print("MPS device not available. Skipping test.")
        return

    device = torch.device("mps")
    dtype = torch.float32

    # Logic derived from tf.keras.backend.moving_average_update
    # x = x * momentum + value * (1 - momentum)
    def moving_average_update(x, value, momentum):
        return x * momentum + value * (1 - momentum)

    # The function to be compiled, combining the update logic with the bug-triggering linear algebra
    def compute_loss(x, matrix, momentum):
        # Apply the moving average update logic
        updated_x = moving_average_update(x, matrix[0], momentum)
        
        # Perform linear algebra operations that triggered the original bug
        # The bug report specifically mentions Cholesky, Inverse, and MatMul
        p_mat_sqrt = torch.linalg.cholesky(matrix).contiguous()
        p_mat_sqrt_inv = p_mat_sqrt.inverse()
        
        # Calculate value using the updated tensor
        val = torch.sum((p_mat_sqrt_inv @ updated_x) ** 2)
        return -val / 2

    # Setup vmap and grad as in the original bug report
    # We map over the first argument (x), while matrix and momentum are broadcasted (None)
    grad_func = torch.func.grad(compute_loss, 0)
    vmap_grad_func = torch.vmap(grad_func, (0, None, None))

    # Compile the function
    compiled_function = torch.compile(vmap_grad_func)

    # Prepare input data
    # x shape: (Batch, Features) -> (2, 3)
    data = torch.randn((2, 3), device=device, dtype=dtype)
    
    # matrix shape: (3, 3) - Positive definite matrix for Cholesky
    p = torch.diag(torch.tensor((20., 0.5, 5.), device=device, dtype=dtype)**2)
    
    momentum = 0.9

    # Execute the test
    try:
        res = compiled_function(data, p, momentum)
        # If successful, check shape
        assert res.shape == (2, 3), f"Expected shape (2, 3), got {res.shape}"
        print("Test Passed: Function compiled and ran successfully on MPS.")
        
    except RuntimeError as e:
        error_msg = str(e)
        if "is_contiguous() INTERNAL ASSERT FAILED" in error_msg:
            print(f"Bug Reproduced: {error_msg}")
            raise AssertionError("MPS contiguity bug detected.")
        else:
            # Re-raise if it's a different error
            raise

if __name__ == "__main__":
    test_mps_compile_contiguity_with_moving_avg()
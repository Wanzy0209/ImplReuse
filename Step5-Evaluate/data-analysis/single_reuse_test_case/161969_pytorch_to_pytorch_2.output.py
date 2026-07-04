import torch

def example_function():
    # Adapted function using torch.prod instead of torch.linalg.cholesky
    def logp(x, matrix):
        # Using torch.prod on a slice of x to test contiguity and compilation behavior
        # similar to the original bug report's context.
        val = torch.prod(x[0, :])
        return -val

    # Keeping the vmap and grad structure as in the original bug report
    score_func = torch.vmap(torch.func.grad(logp, 0), (0, None))

    return score_func

if __name__ == "__main__":
    # Check for MPS availability
    if not torch.backends.mps.is_available():
        print("MPS is not available. Skipping test.")
        exit()

    device = torch.device("mps")
    dtype = torch.float32
    
    # Create input data
    data = torch.ones((2, 5, 3), device=device, dtype=dtype) * 2.0
    compiled_function = torch.compile(example_function())

    p = torch.diag(torch.tensor((20., 0.5, 5,), device=device, dtype=dtype)**2)
    
    try:
        # Run the compiled function
        res = compiled_function(data, p)
        
        # Assertions to verify correctness
        assert res is not None, "Result is None"
        assert res.shape == (2, 5, 3), f"Expected shape (2, 5, 3), got {res.shape}"
        assert torch.isfinite(res).all(), "Result contains NaN or Inf"
        
        print("Test passed successfully.")
    except RuntimeError as e:
        print(f"Test failed with RuntimeError: {e}")
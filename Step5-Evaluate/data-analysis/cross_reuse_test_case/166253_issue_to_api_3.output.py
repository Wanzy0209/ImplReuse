import torch

def test_scaled_modified_bessel_k1_compile_float64():
    """
    Test case for torch.special.scaled_modified_bessel_k1 based on the 
    torch.full bug (Issue 166253).
    
    The original bug involved torch.compile caching the value argument 
    incorrectly for float64 dtypes on CPU. This test checks if 
    scaled_modified_bessel_k1 exhibits similar behavior where the input 
    value is ignored in subsequent compiled calls.
    """
    
    # Define the function using the similar API
    def func_nojit(x):
        return torch.special.scaled_modified_bessel_k1(x)

    # Compile the function
    func_jit = torch.compile(func_nojit)

    # Create inputs with dtype=torch.float64, matching the bug report's condition
    x1 = torch.tensor(5.0, dtype=torch.float64)
    x2 = torch.tensor(10.0, dtype=torch.float64)

    # Get results from non-compiled function
    res_nojit_1 = func_nojit(x1)
    res_nojit_2 = func_nojit(x2)

    # Get results from compiled function
    res_jit_1 = func_jit(x1)
    res_jit_2 = func_jit(x2)

    # 1. Verify non-compiled results are different (sanity check)
    assert not torch.allclose(res_nojit_1, res_nojit_2), \
        "Non-compiled results for different inputs should differ."

    # 2. Verify compiled results are different (Bug reproduction check)
    # The original bug caused the second call to return the result of the first call.
    assert not torch.allclose(res_jit_1, res_jit_2), \
        "Compiled results for different inputs should differ. " \
        "If this fails, the input value might be cached incorrectly like in torch.full."

    # 3. Verify correctness
    assert torch.allclose(res_jit_1, res_nojit_1), \
        "Compiled result for x1 does not match non-compiled result."
    assert torch.allclose(res_jit_2, res_nojit_2), \
        "Compiled result for x2 does not match non-compiled result."

if __name__ == "__main__":
    test_scaled_modified_bessel_k1_compile_float64()
    print("Test passed.")
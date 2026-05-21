import torch

def test_rand_like_with_dtype_in_compile():
    """
    Test case for torch.rand_like with dtype argument under torch.compile.
    This mirrors the logic of the reported issue for torch.mm (Issue 163275),
    where an argument specifying the output data type caused a failure.
    """
    # Setup input tensor matching the original bug report's context (CUDA, float16)
    A = torch.rand((1024, 1024), device="cuda", dtype=torch.float16)

    @torch.compile
    def func(input):
        # torch.rand_like is the similar API. 
        # It accepts a 'dtype' argument to change the output type, 
        # analogous to 'out_dtype' in torch.mm.
        return torch.rand_like(input, dtype=torch.float32)

    # Execute the compiled function
    result = func(A)
    
    # Assertions to verify correctness
    assert result.dtype == torch.float32, f"Expected float32, got {result.dtype}"
    assert result.shape == A.shape, f"Shape mismatch: {result.shape} vs {A.shape}"
    assert result.device == A.device, f"Device mismatch: {result.device} vs {A.device}"

if __name__ == "__main__":
    test_rand_like_with_dtype_in_compile()
    print("Test passed.")
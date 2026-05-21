import torch

def test_torch_mm_out_dtype_compile():
    """
    Test case to verify that torch.mm handles the out_dtype argument
    correctly when used with torch.compile.
    """
    # Check for CUDA availability as the original issue was specific to CUDA
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test.")
        return

    # Define inputs with float16 dtype
    A = torch.rand((1024, 1024), device="cuda", dtype=torch.float16)
    B = torch.rand((1024, 1024), device="cuda", dtype=torch.float16)

    # Define the compiled function using torch.mm with out_dtype
    @torch.compile
    def mm_func(input, weight):
        return torch.mm(input, weight, out_dtype=torch.float32)

    # Execute the function
    result = mm_func(A, B)

    # Assertion to verify the output dtype matches the requested out_dtype
    assert result.dtype == torch.float32, f"Expected torch.float32, but got {result.dtype}"
    
    # Verify the result is on the correct device
    assert result.device == A.device, f"Expected device {A.device}, but got {result.device}"

    print("Test passed: torch.mm with out_dtype works inside torch.compile.")

if __name__ == "__main__":
    test_torch_mm_out_dtype_compile()
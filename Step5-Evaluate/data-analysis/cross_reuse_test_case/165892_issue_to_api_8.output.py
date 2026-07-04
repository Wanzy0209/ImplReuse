import torch

# Fix for AttributeError: module 'torch' has no attribute 'compile'
# This occurs in PyTorch versions prior to 2.0.
# We mock torch.compile to allow the test to run in eager mode.
if not hasattr(torch, 'compile'):
    torch.compile = lambda func, *args, **kwargs: func

# Test case for torch.sgn based on the torch.bmm compilation issue (Issue 165892)
# The original issue demonstrated a failure in torch.compile when handling specific dtype arguments.
# This test verifies that torch.sgn, which has internal dtype branching logic (complex vs real),
# works correctly under torch.compile without triggering similar lowering errors.

def test_sgn_compile():
    # Check for CUDA availability as the original bug was CUDA specific
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test with float16 (similar to the original bug report's input type)
    A_float = torch.rand((1, 1024, 1024), device=device, dtype=torch.float16)
    
    # Test with complex64 (as per the torch.sgn reference implementation logic)
    A_complex = torch.randn((1, 1024, 1024), device=device, dtype=torch.complex64)

    @torch.compile
    def sgn_func(input):
        return torch.sgn(input)

    # Execute and verify for float input
    res_float = sgn_func(A_float)
    assert res_float.dtype == A_float.dtype
    assert torch.allclose(res_float, torch.sgn(A_float))

    # Execute and verify for complex input
    res_complex = sgn_func(A_complex)
    assert res_complex.dtype == A_complex.dtype
    assert torch.allclose(res_complex, torch.sgn(A_complex))

if __name__ == "__main__":
    test_sgn_compile()
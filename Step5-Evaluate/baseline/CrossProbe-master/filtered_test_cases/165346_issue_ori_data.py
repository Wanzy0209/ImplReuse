import torch

# Reproduce the failing test case
if torch.cuda.is_available() and torch.version.hip:
    device = "cuda:0"
    dtype = torch.complex64
    
    # Create test input similar to the failing case
    input_tensor = torch.randn(2, 5, 5, device=device, dtype=dtype)
    args = [torch.randn(2, 5, 5, device=device, dtype=dtype, contiguous=False)]
    
    # Test the operations that are failing
    expected = torch.linalg.cholesky_solve(input_tensor, args[0])
    with_mathview = torch.conj(input_tensor)
    with_mathview = torch.linalg.cholesky_solve(with_mathview, args[0])
    
    # This should fail on ROCm
    torch.testing.assert_close(expected, with_mathview, atol=1e-5, rtol=1.3e-6)
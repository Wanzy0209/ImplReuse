import torch

def test_float64_matmul_rocm_crash():
    """
    Test case for Issue 161256: Simple float64 Matmul on RX 7600xt with ROCm 6.4.3 crashes.
    
    This test leverages torch.backends.cuda.is_built to ensure the environment 
    supports the CUDA backend (which includes ROCm) before attempting the 
    operation that causes the segmentation fault.
    """
    
    # Leverage the similar API to verify the backend is built.
    # The bug is specific to the ROCm/CUDA backend implementation.
    if not torch.backends.cuda.is_built():
        print("Skipping test: PyTorch is not built with CUDA/ROCm support.")
        return

    # Ensure a device is actually available
    if not torch.cuda.is_available():
        print("Skipping test: No CUDA/ROCm device available.")
        return

    # Original bug reproduction logic
    torch.set_default_dtype(torch.float64)
    x = torch.randn(1000, 1000, device="cuda")
    y = torch.randn(1000, 1000, device="cuda")
    
    # The operation causing the segfault in the reported version
    z = x @ y
    
    # Assertions to verify the operation completed successfully without crashing
    assert z.shape == (1000, 1000), "Output shape mismatch"
    assert z.dtype == torch.float64, "Output dtype mismatch"
    
    # Access element to ensure computation is materialized
    val = z[0, 0].item()
    assert isinstance(val, float), "Output value is not a float"
    
    print("Test passed: float64 matmul executed successfully on CUDA/ROCm.")

if __name__ == "__main__":
    test_float64_matmul_rocm_crash()
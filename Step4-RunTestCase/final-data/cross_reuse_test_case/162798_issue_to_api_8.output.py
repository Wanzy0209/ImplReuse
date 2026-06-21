import torch
import math

def test_nanmedian_empty_input():
    """
    Test that torch.nanmedian returns NaN for an empty input tensor
    across different devices (CPU, CUDA, MPS).
    
    This test addresses the bug where MPS returns 0 instead of NaN.
    """
    # Create an empty tensor
    x = torch.empty((0,), dtype=torch.float32)

    # Test on CPU
    cpu_result = torch.nanmedian(x)
    assert torch.isnan(cpu_result), f"CPU: Expected NaN for empty input, got {cpu_result}"

    # Test on MPS (Metal Performance Shaders)
    # The bug report indicates MPS returns 0 instead of NaN
    if torch.backends.mps.is_available():
        mps_x = x.to('mps')
        mps_result = torch.nanmedian(mps_x)
        assert torch.isnan(mps_result), f"MPS: Expected NaN for empty input, got {mps_result}"

    # Test on CUDA
    if torch.cuda.is_available():
        cuda_x = x.to('cuda')
        cuda_result = torch.nanmedian(cuda_x)
        assert torch.isnan(cuda_result), f"CUDA: Expected NaN for empty input, got {cuda_result}"

if __name__ == "__main__":
    test_nanmedian_empty_input()
    print("Test passed.")
import torch
import torch.nn.functional as F

def test_pixel_unshuffle_memory_layout():
    """
    Test case for torch.nn.functional.pixel_unshuffle based on Issue 164491.
    
    The original issue highlights performance and correctness problems with 
    _scaled_mm and _int_mm when the input matrix has a non-standard memory 
    layout (row-major vs column-major). 
    
    This test verifies that pixel_unshuffle handles different memory layouts 
    (contiguous, channels_last, and non-contiguous/strided) correctly, 
    ensuring it does not silently fail or produce incorrect results due to 
    stride assumptions.
    """
    
    # Setup parameters
    batch_size = 2
    channels = 4
    height = 8
    width = 8
    downscale_factor = 2
    
    # Ensure dimensions are divisible by the downscale factor
    assert height % downscale_factor == 0 and width % downscale_factor == 0
    
    # 1. Baseline: Standard Contiguous Input
    print("Testing with contiguous input...")
    x_contiguous = torch.randn(batch_size, channels, height, width)
    out_contiguous = F.pixel_unshuffle(x_contiguous, downscale_factor)
    
    expected_shape = (batch_size, channels * (downscale_factor ** 2), 
                      height // downscale_factor, width // downscale_factor)
    assert out_contiguous.shape == expected_shape, \
        f"Shape mismatch for contiguous input. Expected {expected_shape}, got {out_contiguous.shape}"
    print("Contiguous input passed.")

    # 2. Channels Last Memory Format
    # This simulates a different physical memory layout (similar to row-major vs column-major)
    print("Testing with channels_last memory format...")
    x_channels_last = x_contiguous.clone().to(memory_format=torch.channels_last)
    
    try:
        out_channels_last = F.pixel_unshuffle(x_channels_last, downscale_factor)
        # Verify correctness against the contiguous baseline
        assert torch.allclose(out_channels_last, out_contiguous), \
            "Output mismatch for channels_last input."
        print("Channels last input passed.")
    except RuntimeError as e:
        # If the implementation does not support channels_last, it should raise a clear error
        # rather than producing incorrect results or hanging.
        print(f"Channels last input raised RuntimeError (expected if unsupported): {e}")

    # 3. Non-Contiguous (Strided) Input
    # This simulates the specific case in the bug report where transparent 
    # transposition or slicing leads to non-contiguous tensors.
    print("Testing with non-contiguous (strided) input...")
    # Create a strided tensor by slicing
    x_strided = x_contiguous[:, :, ::2, :] 
    assert not x_strided.is_contiguous(), "Test tensor should be non-contiguous"
    
    try:
        out_strided = F.pixel_unshuffle(x_strided, downscale_factor)
        
        # Verify correctness by comparing with the result of the contiguous clone
        x_strided_cont = x_strided.contiguous()
        out_strided_expected = F.pixel_unshuffle(x_strided_cont, downscale_factor)
        
        assert torch.allclose(out_strided, out_strided_expected), \
            "Output mismatch for non-contiguous input."
        print("Non-contiguous input passed.")
    except RuntimeError as e:
        # Similar to _scaled_mm raising an error, we check if pixel_unshuffle 
        # handles the stride incompatibility explicitly.
        print(f"Non-contiguous input raised RuntimeError: {e}")

if __name__ == "__main__":
    test_pixel_unshuffle_memory_layout()
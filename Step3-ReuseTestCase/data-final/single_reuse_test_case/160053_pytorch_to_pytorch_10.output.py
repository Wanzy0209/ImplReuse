import torch
import torch.nn.functional as F

def test_grid_sample_circular_padding_4d():
    """
    Test case adapted from Issue 160053.
    Verifies torch.nn.functional.grid_sample behavior with 4D input 
    and padding_mode="circular", mirroring the original bug report for F.pad.
    """
    # Create a 4D input tensor (Batch, Channel, Height, Width)
    # Original bug report used (2, 2, 2, 2)
    input_tensor = torch.empty(2, 2, 2, 2)
    
    # grid_sample requires a grid tensor of shape (N, H_out, W_out, 2)
    # We create a simple grid that samples the center of the image (normalized coords -1 to 1)
    # For a 2x2 input, the center is roughly at 0,0
    grid = torch.zeros(2, 2, 2, 2) 
    
    # Attempt to call grid_sample with padding_mode="circular"
    # Note: grid_sample also requires an interpolation 'mode', defaulting to 'bilinear'
    try:
        output = F.grid_sample(input_tensor, grid, mode='bilinear', padding_mode='circular', align_corners=True)
        print("grid_sample with padding_mode='circular' succeeded.")
        # If it succeeds, verify output shape matches expected dimensions
        assert output.shape == input_tensor.shape, f"Shape mismatch: expected {input_tensor.shape}, got {output.shape}"
    except NotImplementedError as e:
        print(f"grid_sample with padding_mode='circular' raised NotImplementedError: {e}")
        # This is expected behavior if circular padding is not implemented for grid_sample
    except Exception as e:
        print(f"grid_sample with padding_mode='circular' raised unexpected error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_grid_sample_circular_padding_4d()
import torch
from torch import nn

def test_convtranspose3d_mps_autocast():
    """
    Test that ConvTranspose3d works with torch.amp.autocast on MPS.
    
    This test leverages the defensive backend checking pattern seen in 
    torch.backends.cusparselt.version (checking is_built/is_available) 
    to ensure the test only runs on supported hardware.
    """
    # Reusing the pattern from torch.backends.cusparselt.version to check 
    # if the backend is available before proceeding.
    if not torch.backends.mps.is_available():
        return

    device = torch.device('mps')

    # Original bug reproduction logic:
    # amp.autocast on MPS previously tried to use FP16 for ConvTranspose3d,
    # which is not supported, causing a RuntimeError.
    # The fix ensures it uses FP32.
    with torch.amp.autocast(device_type=device.type):
        m = nn.ConvTranspose3d(16, 33, 3, stride=2)
        m.to(device)
        x = torch.randn(20, 16, 10, 50, 100).to(device)
        u = m(x)

    # Assertions to verify the operation completed successfully
    assert u is not None
    # Expected output size calculation:
    # H_out = (H_in - 1)*stride - 2*padding + dilation*(kernel_size-1) + output_padding + 1
    # Default padding=0, dilation=1, output_padding=0
    # (10-1)*2 + 3 - 1 = 18 + 2 = 21
    # (50-1)*2 + 3 - 1 = 98 + 2 = 101
    # (100-1)*2 + 3 - 1 = 198 + 2 = 201
    assert u.shape == torch.Size([20, 33, 21, 101, 201])
    
    # Ensure no NaNs are present, which might indicate incorrect dtype handling
    assert not torch.isnan(u).any()
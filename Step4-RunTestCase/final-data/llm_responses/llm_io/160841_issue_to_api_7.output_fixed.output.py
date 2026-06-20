import torch
import pytest

def test_cdist_macos_bf16_precision():
    """
    Test case derived from Issue 160841.
    
    The original issue reported that running models on MacOS (MPS/CPU) with 
    automatic dtype selection resulted in garbage output, which was fixed by 
    explicitly using bfloat16 (bf16).
    
    This test adapts that logic to torch.cdist to ensure that distance calculations
    on MacOS backends do not produce garbage (NaNs/Infs) when using bfloat16,
    and that results remain numerically consistent with float32.
    """
    # Determine device: prioritize MPS (MacOS GPU), fallback to CPU (also affected in issue)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Create input tensors
    # Using a batch size and dimension that might stress precision
    batch_size = 4
    dim = 128
    x1 = torch.randn(batch_size, 10, dim, device=device)
    x2 = torch.randn(batch_size, 15, dim, device=device)

    # Calculate reference distance in float32
    dist_fp32 = torch.cdist(x1, x2, p=2.0)

    # Check bfloat16 support on the device
    # The issue fix involved switching to bf16, so we verify cdist works with it.
    if device.type == 'cpu' or (device.type == 'mps' and torch.is_bf16_supported()):
        x1_bf16 = x1.to(torch.bfloat16)
        x2_bf16 = x2.to(torch.bfloat16)

        try:
            # Run cdist with bfloat16 inputs
            dist_bf16 = torch.cdist(x1_bf16, x2_bf16, p=2.0)

            # 1. Check for "garbage" output (NaNs or Infs) as reported in the bug
            assert not torch.isnan(dist_bf16).any(), "cdist returned NaNs (garbage) with bfloat16 on MacOS"
            assert not torch.isinf(dist_bf16).any(), "cdist returned Infs (garbage) with bfloat16 on MacOS"

            # 2. Verify numerical stability: bf16 results should be close to fp32
            # bfloat16 has lower precision, so we use a relative tolerance
            assert torch.allclose(dist_bf16.float(), dist_fp32, rtol=1e-2, atol=1e-5), \
                "cdist bfloat16 results are not consistent with float32 reference"
        except RuntimeError as e:
            # Handle cases where the specific PyTorch build/device does not implement cdist for BFloat16
            if "not implemented" in str(e):
                pytest.skip(f"cdist not implemented for BFloat16 on this device: {e}")
            else:
                raise
    else:
        pytest.skip("bfloat16 not supported on this device, skipping fix verification")

if __name__ == "__main__":
    test_cdist_macos_bf16_precision()
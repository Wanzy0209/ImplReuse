import torch
import pytest

def test_absolute_nan_handling_mps_vs_cpu():
    """
    Test case adapted from Issue 163851 (grid_sampler_3d NaN handling).
    Verifies that torch.absolute handles NaN values consistently between CPU and MPS backends.
    """
    if not torch.backends.mps.is_available():
        pytest.skip("MPS backend not available")

    # Create input tensor containing NaN, similar to the grid_nan in the original issue
    input_nan = torch.tensor([torch.nan, 1.0, -1.0, 0.0])

    # Execute on CPU
    out_cpu = torch.absolute(input_nan)

    # Execute on MPS
    out_mps = torch.absolute(input_nan.to("mps"))

    # Check if NaN is preserved on CPU
    assert torch.isnan(out_cpu[0]), "CPU output should be NaN for NaN input"

    # Check if NaN is preserved on MPS (Original bug was MPS returning 1.0 instead of NaN)
    assert torch.isnan(out_mps[0]), "MPS output should be NaN for NaN input"

    # Verify other values are consistent
    assert torch.allclose(out_cpu[1:], out_mps.cpu()[1:]), "Non-NaN values should match"
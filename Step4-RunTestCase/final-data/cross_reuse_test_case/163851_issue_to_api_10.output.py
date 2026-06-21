import torch
import pytest

def test_grid_sampler_3d_nan_mps():
    """
    Test that grid_sampler_3d correctly handles NaN values in the grid on the MPS backend.
    
    This test reproduces the issue where the MPS backend was clamping NaN values 
    to -1 (resulting in valid output) instead of propagating the NaN like the CPU backend.
    
    The relationship to the similar API (torch.backends.cuda.enable_flash_sdp) 
    is conceptual: both involve backend-specific kernel behavior and configuration.
    Here we verify the correctness of the MPS kernel implementation for grid_sampler_3d.
    """
    if not torch.backends.mps.is_available():
        pytest.skip("MPS backend not available")

    # Setup input and grid with NaN as described in the bug report
    input = torch.ones(1, 1, 3, 3, 3)
    grid_nan = torch.tensor([[[[[torch.nan, 1., 1.], [1., 1., 1.]]]]])

    # Compute on CPU (Expected behavior)
    out_cpu = torch.grid_sampler_3d(input, grid_nan, 0, 0, True)
    
    # Compute on MPS (Device Under Test)
    input_mps = input.to("mps")
    grid_mps = grid_nan.to("mps")
    out_mps = torch.grid_sampler_3d(input_mps, grid_mps, 0, 0, True)

    # Assert that MPS output matches CPU output.
    # Specifically, the first element should be NaN, not 1.0.
    # torch.testing.assert_close handles NaN equality checks (NaN == NaN is True).
    torch.testing.assert_close(out_mps.cpu(), out_cpu)

    # Explicit check for the specific bug case mentioned in the issue
    assert torch.isnan(out_cpu.flatten()[0]), "CPU output should start with NaN"
    assert torch.isnan(out_mps.flatten()[0]), "MPS output should start with NaN (Bug fix verification)"
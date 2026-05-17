import torch
import torch.nn.functional as F

def test_interpolate_3d_mps():
    """
    Test case for torch.nn.functional.interpolate on MPS device.
    This test is derived from a bug report where 'aten::grid_sampler_3d' 
    was not implemented for MPS. We verify if the similar API, interpolate,
    supports 3D (volumetric) operations on the MPS backend.
    """
    # Check if MPS is available, as the bug is specific to Mac Metal (MPS)
    if not torch.backends.mps.is_available():
        print("MPS device is not available. Skipping test.")
        return

    device = torch.device("mps")

    # Create a 5D input tensor (Batch, Channels, Depth, Height, Width)
    # This mimics the 3D nature of the failing 'grid_sampler_3d' operation
    input_tensor = torch.randn(1, 3, 8, 16, 16, device=device)

    # Perform 3D interpolation using 'trilinear' mode
    # 'trilinear' is the 3D equivalent of the interpolation modes often used with grid_sample
    try:
        output_tensor = F.interpolate(
            input_tensor,
            size=(16, 32, 32),  # Upsample depth, height, and width
            mode='trilinear',
            align_corners=False
        )

        # Verify the output is on the correct device and has the expected shape
        assert output_tensor.device == device, "Output tensor is not on MPS device"
        assert output_tensor.shape == (1, 3, 16, 32, 32), f"Output shape mismatch: {output_tensor.shape}"

        print("Test Passed: torch.nn.functional.interpolate works on MPS for 3D tensors.")

    except NotImplementedError as e:
        print(f"Test Failed: {e}")

if __name__ == "__main__":
    test_interpolate_3d_mps()
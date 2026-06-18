import torch

def test_torch_device_mps():
    """
    Test case for torch.device with MPS backend.
    Adapted from Issue #163721 to verify device creation and basic tensor operations.
    """
    # Check if MPS backend is available
    if not torch.backends.mps.is_available():
        print("MPS backend is not available. Skipping test.")
        return

    # Original API call site: torch.device("mps")
    mps_device = torch.device("mps")

    # Verify the device type
    assert mps_device.type == "mps", f"Expected device type 'mps', got '{mps_device.type}'"

    # Verify basic tensor creation and operation on the device
    # This ensures the device context is valid and doesn't segfault immediately upon use
    try:
        x = torch.randn(10, 10, device=mps_device)
        y = torch.randn(10, 10, device=mps_device)
        
        # Perform a simple operation
        z = x + y
        
        # Verify the result is on the correct device
        assert z.device.type == "mps", "Result tensor is not on MPS device"
        
    except RuntimeError as e:
        print(f"RuntimeError during MPS operation: {e}")
        raise

if __name__ == "__main__":
    test_torch_device_mps()
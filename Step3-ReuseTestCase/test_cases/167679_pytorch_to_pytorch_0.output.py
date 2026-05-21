import torch
import platform
import sys

def test_mps_availability_on_macos():
    """
    Test case to verify MPS availability on macOS Apple Silicon.
    Addresses Issue 167679 where is_built() returns True but is_available() returns False.
    """
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Platform: {platform.platform()}")
    print(f"Arch: {platform.machine()}")

    is_built = torch.backends.mps.is_built()
    is_available = torch.backends.mps.is_available()

    print(f"MPS built?: {is_built}")
    print(f"MPS available?: {is_available}")

    # Check if we are on macOS Apple Silicon
    is_macos = platform.system() == "Darwin"
    is_apple_silicon = platform.machine() == "arm64"

    if is_macos and is_apple_silicon:
        # If MPS backend is compiled in the binary, it should be available on Apple Silicon
        # (assuming the OS version supports it, which is checked internally by is_available).
        # The bug report indicates is_built=True and is_available=False on macOS 26.
        if is_built:
            assert is_available, (
                f"Bug 167679 Regression: MPS is built ({is_built}) but not available ({is_available}) "
                f"on {platform.platform()}. Expected is_available to be True."
            )
        else:
            print("MPS is not built in this installation. Skipping availability assertion.")
    else:
        print("Not running on macOS Apple Silicon. Skipping MPS availability assertion.")

if __name__ == "__main__":
    test_mps_availability_on_macos()
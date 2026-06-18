import torch
import platform

# The similar API tf.compat.forward_compatibility_horizon is used to verify
# behavior against specific version constraints. We apply this pattern here
# to test the MPS backend availability check against the macOS 13.0+ requirement.

def test_mps_version_requirement():
    """
    Tests if the MPS backend correctly identifies macOS 13.0+ (e.g., 13.7.4)
    as a supported version.
    """
    # Check if we are on the target platform
    if platform.system() != "Darwin":
        print("Skipping test: Not on macOS.")
        return

    # Reproduce the bug logic
    try:
        # Attempt to move a tensor to the MPS device
        foo = torch.tensor([[1]])
        foo = foo.to('mps')
        print("Test Passed: MPS backend is available.")
    except RuntimeError as e:
        error_message = str(e)
        # The bug report shows this specific error message appearing incorrectly
        # on valid versions like 13.7.4
        if "MacOS 13.0+" in error_message:
            print(f"Test Failed: Incorrect version rejection on {platform.mac_ver()[0]}.")
            print(f"Details: {error_message}")
            raise AssertionError("MPS backend version check logic is incorrect.")
        else:
            # Re-raise if it's a different MPS error (e.g., no GPU)
            raise

if __name__ == "__main__":
    test_mps_version_requirement()
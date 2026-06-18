import torch
import platform
import sys
import re

def test_mps_backend_version_check():
    """
    Test case to verify MPS backend availability logic on macOS.
    
    This test preserves the original bug reproduction logic (tensor.to('mps'))
    while leveraging the state-checking pattern observed in the similar API
    (tf.keras.backend.get_uid), where a condition (graph existence / OS version)
    is explicitly verified before proceeding with the operation.
    """
    # Check if we are on macOS
    if sys.platform != 'darwin':
        print("Skipping test: Not on macOS")
        return

    # Get the macOS version
    # The bug report suggests using `sw_vers`, we use platform.mac_ver() for Pythonic access.
    release = platform.mac_ver()[0]
    if not release:
        print("Skipping test: Could not determine macOS version")
        return

    # Parse version string (e.g., "13.7.4") to extract the major version
    try:
        major_version = int(re.split(r'[.]', release)[0])
    except ValueError:
        print(f"Skipping test: Could not parse version string '{release}'")
        return

    # Original bug reproduction logic
    foo = torch.tensor([[1]])

    # State check pattern: Verify OS version condition before asserting behavior
    # Similar to how tf.keras.backend.get_uid checks 'if graph not in PER_GRAPH_OBJECT_NAME_UIDS'
    if major_version >= 13:
        try:
            mps_foo = foo.to('mps')
            # If successful, verify the device type
            assert mps_foo.device.type == 'mps', "Tensor did not move to MPS device"
        except RuntimeError as e:
            # The bug: RuntimeError claiming 13.0+ required on macOS 13.7.4
            if "13.0+" in str(e) and "MacOS 13.0+" in str(e):
                raise AssertionError(
                    f"Bug reproduced: MPS backend incorrectly unavailable on macOS {release} (>= 13.0). "
                    f"The version check logic seems to be failing (e.g. string comparison issue)."
                ) from e
            # Other RuntimeErrors (e.g., driver not installed) are acceptable for this specific logic test
            print(f"Note: MPS failed with a different error (likely hardware/driver): {e}")
    else:
        # For versions < 13.0, the error is expected
        try:
            mps_foo = foo.to('mps')
            print(f"Warning: MPS available on macOS {release} (< 13.0), which is unexpected but not the reported bug.")
        except RuntimeError as e:
            if "13.0+" in str(e):
                # Expected behavior for older OS versions
                pass
            else:
                raise

if __name__ == "__main__":
    test_mps_backend_version_check()